use core::future::Future;

use std::collections::HashMap;

use gc::{Gc, GcCell, Trace, Finalize, custom_trace};
use gc_derive::{Trace, Finalize};

mod event_loop;
use event_loop::EventLoop;

type InstructionIndex = usize;
type GlobalIndex = usize;
type LocalIndex = usize;
type ScopeIndex = usize;
type AncestorDistance = usize;
type Arity = usize; // ranges between zero and fifteen
type AsyncId = usize;

/// A value of the virtual machine.
pub trait Value: Sized + Trace + Finalize + Clone + Default + 'static {
    type Failure; // The type of reasons for why a built-in function can abort the computation
    type BuiltInFunction: BuiltInSynchronousFunction<Value = Self, Failure = Self::Failure>;
    type BuiltInAsync: BuiltInAsyncFunction<Value = Self, Failure = Self::Failure>;

    fn truthy(&self) -> bool;

    fn as_built_in_function(self) -> Option<Self::BuiltInFunction>;
    fn as_built_in_function_ref(&self) -> Option<&Self::BuiltInFunction>;
    fn as_built_in_function_mut(&mut self) -> Option<&mut Self::BuiltInFunction>;

    fn as_built_in_async(self) -> Option<Self::BuiltInAsync>;
    fn as_built_in_async_ref(&self) -> Option<&Self::BuiltInAsync>;
    fn as_built_in_async_mut(&mut self) -> Option<&mut Self::BuiltInAsync>;

    fn new_function(f: Function<Self>) -> Self;
    fn as_function(self) -> Option<Function<Self>>;
    fn as_function_ref(&self) -> Option<&Function<Self>>;
    fn as_function_mut(&mut self) -> Option<&mut Function<Self>>;
}

/// A non-built-in function.
#[derive(Trace, Finalize)]
pub struct Function<V: Value> {
  ordinal: usize,
  parent_scope: Option<Gc<GcCell<Scope<V>>>>,
  asynchronous: bool,
  arity: Arity,
  local: LocalIndex,
  scoped: ScopeIndex,
  instructions: InstructionIndex,
}

pub trait BuiltInSynchronousFunction {
    type Value;
    type Failure;
    fn arity(&self) -> Arity;
    fn invoke(&mut self, args: &[Self::Value]) -> Result<Self::Value, Self::Failure>;
}

pub trait BuiltInAsyncFunction {
    type Value;
    type Failure;
    type F: Future<Output = Result<Self::Value, Self::Failure>>;
    fn arity(&self) -> Arity;
    fn invoke(&mut self, args: &[Self::Value]) -> Self::F;
}

pub struct VirtualMachine<V: Value> {
    instructions: Box<[Instruction]>,
    instruction_counter: InstructionIndex,
    globals: Box<[V]>,
    stack: Vec<StackFrame<V>>,
    asyncs: HashMap<AsyncId, AsyncFrame<V>>,
    current_async: AsyncId,
    pending_stack: Vec<(AsyncId, InstructionIndex)>,
    event_loop: EventLoop<V>,
    next_async_id: AsyncId,
    next_ordinal: usize,
}

#[derive(Finalize)]
pub struct Scope<V: Value> {
    parent: Option<Gc<GcCell<Scope<V>>>>,
    values: Box<[V]>,
}

unsafe impl<V: Value> Trace for Scope<V> {
    custom_trace!(this, {
        mark(&this.parent);
        mark(&this.values);
    });
}

struct StackFrame<V: Value> {
  return_instruction: InstructionIndex,
  dst: Address,
  scope: Gc<GcCell<Scope<V>>>,
  values: Box<[V]>,
}

struct AsyncFrame<V: Value> {
    scope: Gc<GcCell<Scope<V>>>,
    values: Box<[V]>,
    continuation: Option<Continuation>,
    pending_strands: usize,
}

#[derive(Clone, Copy)]
struct Continuation {
    instruction: InstructionIndex,
    dst: Address,
    call: AsyncId,
}

#[derive(Clone, Copy)]
pub enum Address {
    Global(GlobalIndex),
    Local(LocalIndex),
    Scoped { up: AncestorDistance, index: ScopeIndex },
}

pub enum Instruction {
    Jump(InstructionIndex),
    ConditionalJump { condition: Address, target: InstructionIndex },
    Assign { src: Address, dst: Address },
    Return(Address),
    CreateFunction {
        dst: Address,
        asynchronous: bool,
        arity: Arity,
        local: LocalIndex,
        scoped: ScopeIndex,
        instructions: InstructionIndex,
    },
    Call {
        dst: Address,
        callee: Address,
        arguments: Box<[Address]>,
    },
    ConcurrentCall {
        dst: Address,
        continue_at: InstructionIndex,
        callee: Address,
        arguments: Box<[Address]>,
        },
    Yield,
}

pub enum Failure<V, C> {
    Arity {
        expected: Arity,
        actual: Arity,
    },
    NotASynchronousFunction(V),
    NotAnAsynchronousFunction(V),
    EmptyEventLoop,
    Custom(C),
}

impl<V, C> From<C> for Failure<V, C> {
    fn from(error: C) -> Self {
        Failure::Custom(error)
    }
}

impl<V: Value> VirtualMachine<V> {
    pub fn new(
        instructions: Box<[Instruction]>,
        globals: Box<[V]>,
        initial_instruction_counter: InstructionIndex,
        initial_scope: Gc<GcCell<Scope<V>>>,
        initial_local_values: Box<[V]>,
    ) -> Self {
        let initial_call = AsyncFrame {
            scope: initial_scope,
            values: initial_local_values,
            continuation: None,
            pending_strands: 0,
        };

        let mut asyncs = HashMap::new();
        asyncs.insert(0, initial_call);

        VirtualMachine {
            instructions,
            instruction_counter: initial_instruction_counter,
            globals,
            stack: vec![],
            asyncs,
            current_async: 0,
            pending_stack: Vec::new(),
            event_loop: EventLoop::new(),
            next_async_id: 1,
            next_ordinal: 0,
        }
    }

    pub fn run(&mut self) -> Result<V, Failure<V, <V as Value>::Failure>> {
        loop {
            match self.instructions.get(self.instruction_counter).unwrap() {
                Instruction::Assign { src, dst } => {
                    let v = self.load(*src);
                    let tmp = *dst; // appeasing the borrow checker
                    self.store(v, tmp);
                    self.instruction_counter += 1;
                }

                Instruction::Jump(target) => self.instruction_counter = *target,

                Instruction::ConditionalJump { condition, target } => {
                    if self.load(*condition).truthy() {
                        self.instruction_counter = *target;
                    } else {
                        self.instruction_counter += 1;
                    }
                }

                Instruction::CreateFunction {
                    dst,
                    asynchronous,
                    arity,
                    local,
                    scoped,
                    instructions,
                } => {
                    let ordinal = self.next_ordinal;
                    self.next_ordinal += 1;

                    let f = V::new_function(Function {
                        ordinal,
                        parent_scope: Some(self.scope().clone()),
                        asynchronous: *asynchronous,
                        arity: *arity,
                        local: *local,
                        scoped: *scoped,
                        instructions: *instructions,
                    });

                    let tmp = *dst; // appeasing the borrow checker
                    self.store(f, tmp);
                    self.instruction_counter += 1;
                }

                Instruction::Call { dst, callee, arguments } => {
                    let mut f = self.load(*callee);
                    let args = self.resolve_arguments(arguments);

                    if let Some(b) = f.as_built_in_function_mut() {
                        if b.arity() != args.len() {
                            return Err(Failure::Arity { expected: b.arity(), actual: args.len() });
                        }

                        let v = b.invoke(&args)?;
                        let tmp = *dst; // appeasing the borrow checker
                        self.store(v, tmp);
                        self.instruction_counter += 1;
                    } else if let Some(Function {
                        ordinal: _,
                        parent_scope,
                        asynchronous,
                        arity,
                        local,
                        scoped,
                        instructions,
                    }) = f.as_function_ref() {
                        if *asynchronous {
                            return Err(Failure::NotASynchronousFunction(f));
                        }

                        if *arity != args.len() {
                            return Err(Failure::Arity {expected: *arity, actual: args.len()});
                        }

                        let locals = prepare_locals(*local, args);
                        let scope = prepare_scope(*scoped, parent_scope.clone());

                        let frame = StackFrame {
                            return_instruction: self.instruction_counter + 1,
                            dst: *dst,
                            scope,
                            values: locals,
                        };

                        self.stack.push(frame);
                        self.instruction_counter = *instructions;
                    } else {
                        return Err(Failure::NotASynchronousFunction(f));
                    }
                }

                Instruction::ConcurrentCall { dst, continue_at, callee, arguments } => {
                    self.assert_asynchronous();
                    let mut f = self.load(*callee);
                    let args = self.resolve_arguments(arguments);

                    if let Some(a) = f.as_built_in_async_mut() {
                        if a.arity() != args.len() {
                            return Err(Failure::Arity { expected: a.arity(), actual: args.len() });
                        }

                        let continuation = Continuation {
                            instruction: *continue_at,
                            dst: *dst,
                            call: self.current_async,
                        };

                        self.event_loop.spawn(a.invoke(&args), continuation);
                        self.instruction_counter += 1;
                    } else if let Some(Function {
                        ordinal: _,
                        parent_scope,
                        asynchronous,
                        arity,
                        local,
                        scoped,
                        instructions,
                    }) = f.as_function_ref() {
                        if !asynchronous {
                            return Err(Failure::NotAnAsynchronousFunction(f));
                        }

                        if *arity != args.len() {
                            return Err(Failure::Arity { expected: *arity, actual: args.len() });
                        }

                        let locals = prepare_locals(*local, args);
                        let scope = prepare_scope(*scoped, parent_scope.clone());

                        let id = self.next_async_id;
                        self.next_async_id += 1;

                        let continuation = Some(Continuation {
                            instruction: *continue_at,
                            dst: *dst,
                            call: self.current_async,
                        });

                        self.asyncs.insert(id, AsyncFrame {
                            scope,
                            values: locals,
                            continuation,
                            pending_strands: 0,
                        });

                        self.pending_stack.push((id, *instructions));

                        self.asyncs.get_mut(&self.current_async).unwrap().pending_strands += 1;

                        self.instruction_counter += 1;
                    } else {
                        return Err(Failure::NotAnAsynchronousFunction(f));
                    }
                }

                Instruction::Return(src) => {
                    let v = self.load(*src);

                    match self.stack.pop() {
                        Some(StackFrame { return_instruction, dst, .. }) => {
                            self.store(v, dst);
                            self.instruction_counter = return_instruction;
                        }
                        None => {
                            match self.asyncs.get(&self.current_async) {
                                None => {}
                                Some(call) => {
                                    match call.continuation {
                                        None => return Ok(v),
                                        Some(continuation) => self.async_return(v, continuation),
                                    }
                                }
                            }
                        }
                    }
                }

                Instruction::Yield => {
                    self.assert_asynchronous();

                    if self.asyncs[&self.current_async].pending_strands == 0 {
                        self.asyncs.remove(&self.current_async);
                    } else {
                        self.asyncs.get_mut(&self.current_async).unwrap().pending_strands -= 1
                    }

                    match self.pending_stack.pop() {
                        Some((call_id, instruction)) => {
                            self.current_async = call_id;
                            self.instruction_counter = instruction;
                        }
                        None => {
                            match smol::block_on(self.event_loop.next()) {
                                None => return Err(Failure::EmptyEventLoop),
                                Some((Err(custom_failure), _)) => return Err(Failure::Custom(custom_failure)),
                                Some((Ok(v), continuation)) => self.async_return(v, continuation),
                            }
                        }
                    }
                }
            }
        }
    }

    fn scope(&self) -> &Gc<GcCell<Scope<V>>> {
        match self.stack.last() {
            Some(frame) => &frame.scope,
            None => &self.asyncs[&self.current_async].scope,
        }
    }

    fn locals(&self) -> &[V] {
        match self.stack.last() {
            Some(frame) => &frame.values,
            None => &self.asyncs[&self.current_async].values,
        }
    }

    fn locals_mut(&mut self) -> &mut [V] {
        match self.stack.last_mut() {
            Some(frame) => &mut frame.values,
            None => &mut self.asyncs.get_mut(&self.current_async).unwrap().values,
        }
    }

    fn assert_asynchronous(&self) {
        assert!(self.stack.len() > 0);
    }

    fn async_return(&mut self, v: V, continuation: Continuation) {
        self.store(v, continuation.dst);
        self.current_async = continuation.call;
        self.instruction_counter = continuation.instruction;
        self.asyncs.remove(&self.current_async);
    }

    fn load(&self, a: Address) -> V {
        match a {
            Address::Global(i) => self.globals[i].clone(),
            Address::Local(i) => self.locals()[i].clone(),
            Address::Scoped { up, index } => load_scoped(self.scope(), up, index),
        }
    }

    fn store(&mut self, v: V, a: Address) {
        match a {
            Address::Global(i) => self.globals[i] = v,
            Address::Local(i) => self.locals_mut()[i] = v,
            Address::Scoped { up, index } => store_scoped(v, self.scope(), up, index),
        }
    }

    fn resolve_arguments(&self, arguments: &[Address]) -> Vec<V> {
        let mut args = vec![];
        for argument in arguments.iter() {
            args.push(self.load(*argument));
        }
        args
    }
}

fn load_scoped<V: Value>(
    scope: &Gc<GcCell<Scope<V>>>,
    up: AncestorDistance,
    index: ScopeIndex,
) -> V {
    if up == 0 {
        scope.borrow().values[index].clone()
    } else {
        load_scoped(scope.borrow().parent.as_ref().unwrap(), up - 1, index)
    }
}

fn store_scoped<V: Value>(
    v: V,
    scope: &Gc<GcCell<Scope<V>>>,
    up: AncestorDistance,
    index: ScopeIndex,
) {
    if up == 0 {
        scope.borrow_mut().values[index] = v
    } else {
        store_scoped(v, scope.borrow().parent.as_ref().unwrap(), up - 1, index)
    }
}

fn prepare_locals<V: Value>(len: LocalIndex, args: Vec<V>) -> Box<[V]> {
    let mut values = Vec::with_capacity(len);
    values.resize_with(len, Default::default);

    for (i, arg) in args.into_iter().enumerate() {
        values[i] = arg;
    }

    return values.into_boxed_slice()
}

fn prepare_scope<V: Value>(len: ScopeIndex, parent_scope: Option<Gc<GcCell<Scope<V>>>>) -> Gc<GcCell<Scope<V>>> {
    let mut scope_values = Vec::with_capacity(len);
    scope_values.resize_with(len, Default::default);

    return Gc::new(GcCell::new(Scope {
        values: scope_values.into_boxed_slice(),
        parent: parent_scope,
    }));
}
