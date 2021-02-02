use core::future::Future;

use std::collections::{VecDeque, HashMap};

use gc::{Gc, GcCell, Trace, Finalize, custom_trace};
use gc_derive::{Trace, Finalize};

mod event_loop;
use event_loop::EventLoop;

type InstructionIndex = usize;
type GlobalIndex = usize;
type LocalIndex = usize;
type ScopeIndex = usize;
type AncestorDistance = usize;
type Arity = usize;
type AsyncCallId = usize;

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
  scope: Gc<GcCell<Scope<V>>>,
  header: InstructionIndex,
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
    instruction_map: Box<[Instruction]>,
    instruction_counter: InstructionIndex,
    global_map: Box<[V]>,
    stack: Vec<StackFrame<V>>,
    active_calls: HashMap<AsyncCallId, AsyncCall<V>>,
    current_call: AsyncCallId,
    pending_queue: VecDeque<(AsyncCallId, InstructionIndex)>,
    return_queue: VecDeque<AsyncReturn<V>>,
    event_loop: EventLoop<V>,
    next_async_id: AsyncCallId,
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

struct AsyncCall<V: Value> {
    scope: Gc<GcCell<Scope<V>>>,
    values: Box<[V]>,
    continuation: Option<Continuation>,
    pending_strands: usize,
}

#[derive(Clone, Copy)]
struct Continuation {
    instruction: InstructionIndex,
    dst: Address,
    call: AsyncCallId,
}

enum AsyncReturn<V> {
    Regular {
        value: V,
        call: AsyncCallId,
    },
    BuiltIn {
        value: V,
        continuation: Continuation,
    }
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
    CreateFunction { dst: Address, header: InstructionIndex },
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
    FunctionHeader {
        asynchronous: bool,
        arity: Arity,
        local: LocalIndex,
        scoped: ScopeIndex,
    },
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
        instruction_map: Box<[Instruction]>,
        global_map: Box<[V]>,
        initial_instruction_counter: InstructionIndex,
        initial_scope: Gc<GcCell<Scope<V>>>,
        initial_local_values: Box<[V]>,
    ) -> Self {
        let initial_call = AsyncCall {
            scope: initial_scope,
            values: initial_local_values,
            continuation: None,
            pending_strands: 0,
        };

        let mut active_calls = HashMap::new();
        active_calls.insert(0, initial_call);

        VirtualMachine {
            instruction_map,
            instruction_counter: initial_instruction_counter,
            global_map,
            stack: vec![],
            active_calls,
            current_call: 0,
            pending_queue: VecDeque::new(),
            return_queue: VecDeque::new(),
            event_loop: EventLoop::new(),
            next_async_id: 1,
            next_ordinal: 0,
        }
    }

    pub fn run(&mut self) -> Result<V, Failure<V, <V as Value>::Failure>> {
        loop {
            match self.instruction_map.get(self.instruction_counter).unwrap() {
                Instruction::FunctionHeader { .. } => self.instruction_counter += 1,

                Instruction::Assign { src, dst } => {
                    store(load(*src, self), *dst, self);
                    self.instruction_counter += 1;
                }

                Instruction::Jump(target) => self.instruction_counter = *target,

                Instruction::ConditionalJump { condition, target } => {
                    if load(*condition, self).truthy() {
                        self.instruction_counter = *target;
                    } else {
                        self.instruction_counter += 1;
                    }
                }

                Instruction::CreateFunction { dst, header } => {
                    if let Instruction::FunctionHeader { scoped, .. } = self.instruction_map[*header] {
                        let ordinal = self.next_ordinal;
                        self.next_ordinal += 1;

                        let mut scope_values = Vec::with_capacity(scoped);
                        scope_values.resize_with(scoped, Default::default);

                        let scope = Gc::new(GcCell::new(Scope {
                            values: scope_values.into_boxed_slice(),
                            parent: Some(self.scope().clone()),
                        }));

                        let f = V::new_function(Function {
                            ordinal,
                            scope,
                            header: *header,
                        });

                        store(f, *dst, self);
                        self.instruction_counter += 1;
                    } else {
                        panic!("Instruction::CreateFunction.header must point to a function header instruction");
                    }
                }

                Instruction::Call { dst, callee, arguments } => {
                    let mut f = load(*callee, self);
                    let args = resolve_arguments(self, arguments);

                    if let Some(b) = f.as_built_in_function_mut() {
                        if b.arity() != args.len() {
                            return Err(Failure::Arity { expected: b.arity(), actual: args.len() });
                        }

                        let v = b.invoke(&args)?;
                        store(v, *dst, self);
                        self.instruction_counter += 1;
                    } else if let Some(Function {scope, header, ..}) = f.as_function_ref() {
                        if let Instruction::FunctionHeader {
                            asynchronous, arity, local, ..
                        } = self.instruction_map[*header] {
                            if asynchronous {
                                return Err(Failure::NotASynchronousFunction(f));
                            }

                            if arity != args.len() {
                                return Err(Failure::Arity {expected: arity, actual: args.len()});
                            }

                            let locals = prepare_locals(local, args);
                            let frame = StackFrame {
                                return_instruction: self.instruction_counter + 1,
                                dst: *dst,
                                scope: scope.clone(),
                                values: locals,
                            };

                            self.stack.push(frame);
                            self.instruction_counter = *header;
                        } else {
                            panic!("Function must point to a function header instruction");
                        }
                    } else {
                        return Err(Failure::NotASynchronousFunction(f));
                    }
                }

                Instruction::ConcurrentCall { dst, continue_at, callee, arguments } => {
                    self.assert_asynchronous();
                    let mut f = load(*callee, self);
                    let args = resolve_arguments(self, arguments);

                    if let Some(a) = f.as_built_in_async_mut() {
                        if a.arity() != args.len() {
                            return Err(Failure::Arity { expected: a.arity(), actual: args.len() });
                        }

                        let continuation = Continuation {
                            instruction: *continue_at,
                            dst: *dst,
                            call: self.current_call,
                        };

                        self.event_loop.spawn(a.invoke(&args), continuation);
                        self.instruction_counter += 1;
                    } else if let Some(Function {scope, header, ..}) = f.as_function_ref() {
                        if let Instruction::FunctionHeader {
                            asynchronous, arity, local, ..
                        } = self.instruction_map[*header] {
                            if !asynchronous {
                                return Err(Failure::NotAnAsynchronousFunction(f));
                            }

                            if arity != args.len() {
                                return Err(Failure::Arity { expected: arity, actual: args.len() });
                            }

                            let locals = prepare_locals(local, args);

                            let id = self.next_async_id;
                            self.next_async_id += 1;

                            let continuation = Some(Continuation {
                                instruction: *continue_at,
                                dst: *dst,
                                call: self.current_call,
                            });

                            self.active_calls.insert(id, AsyncCall {
                                scope: scope.clone(),
                                values: locals,
                                continuation,
                                pending_strands: 0,
                            });

                            self.pending_queue.push_back((id, *header));

                            self.active_calls.get_mut(&self.current_call).unwrap().pending_strands += 1;

                            self.instruction_counter += 1;
                        } else {
                            panic!("Function must point to a function header instruction");
                        }
                    } else {
                        return Err(Failure::NotAnAsynchronousFunction(f));
                    }
                }

                Instruction::Return(src) => {
                    let v = load(*src, self);

                    match self.stack.pop() {
                        Some(StackFrame { return_instruction, dst, .. }) => {
                            store(v, dst, self);
                            self.instruction_counter = return_instruction;
                        }
                        None => {
                            self.return_queue.push_back(AsyncReturn::Regular {
                                value: v,
                                call: self.current_call,
                            });
                            match self.do_yield() {
                                Status::Done(v) => return Ok(v),
                                Status::Nope(failure) => return Err(failure),
                                Status::Continue => {}
                            }
                        }
                    }
                }

                Instruction::Yield => {
                    match self.do_yield() {
                        Status::Done(v) => return Ok(v),
                        Status::Nope(failure) => return Err(failure),
                        Status::Continue => {}
                    }
                }
            }
        }
    }

    fn do_yield(&mut self) -> Status<V, Failure<V, <V as Value>::Failure>> {
        self.assert_asynchronous();

        if self.active_calls[&self.current_call].pending_strands == 0 {
            self.active_calls.remove(&self.current_call);
        } else {
            self.active_calls.get_mut(&self.current_call).unwrap().pending_strands -= 1
        }

        match self.return_queue.pop_front() {
            Some(AsyncReturn::Regular { value, call: call_id }) => {
                match self.active_calls.get(&call_id) {
                    None => return Status::Continue,
                    Some(call) => {
                        match call.continuation {
                            None => return Status::Done(value),
                            Some(Continuation { instruction, call: next_call_id, dst }) => {
                                store(value, dst, self);
                                self.current_call = next_call_id;
                                self.instruction_counter = instruction;
                            }
                        }
                    }
                }

                self.active_calls.remove(&call_id);
                return Status::Continue;
            }
            Some(AsyncReturn::BuiltIn { value, continuation }) => {
                store(value, continuation.dst, self);
                self.current_call = continuation.call;
                self.instruction_counter = continuation.instruction;
                return Status::Continue;
            }
            None => {
                match self.pending_queue.pop_front() {
                    Some((call_id, instruction)) => {
                        self.current_call = call_id;
                        self.instruction_counter = instruction;
                        return Status::Continue;
                    }
                    None => {
                        match smol::block_on(self.event_loop.next()) {
                            None => return Status::Nope(Failure::EmptyEventLoop),
                            Some((Err(custom_failure), _)) => return Status::Nope(Failure::Custom(custom_failure)),
                            Some((Ok(v), continuation)) => {
                                self.return_queue.push_back(AsyncReturn::BuiltIn {
                                    value: v,
                                    continuation,
                                });
                                return Status::Continue;
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
            None => &self.active_calls[&self.current_call].scope,
        }
    }

    fn locals(&self) -> &[V] {
        match self.stack.last() {
            Some(frame) => &frame.values,
            None => &self.active_calls[&self.current_call].values,
        }
    }

    fn locals_mut(&mut self) -> &mut [V] {
        match self.stack.last_mut() {
            Some(frame) => &mut frame.values,
            None => &mut self.active_calls.get_mut(&self.current_call).unwrap().values,
        }
    }

    fn assert_asynchronous(&self) {
        assert!(self.stack.len() > 0);
    }
}

enum Status<V, F> {
    Done(V),
    Nope(F),
    Continue,
}

fn load<V: Value>(a: Address, vm: &VirtualMachine<V>) -> V {
    match a {
        Address::Global(i) => vm.global_map[i].clone(),
        Address::Local(i) => vm.locals()[i].clone(),
        Address::Scoped { up, index } => load_scoped(vm.scope(), up, index),
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

fn store<V: Value>(v: V, a: Address, vm: &mut VirtualMachine<V>) {
    match a {
        Address::Global(i) => vm.global_map[i] = v,
        Address::Local(i) => vm.locals_mut()[i] = v,
        Address::Scoped { up, index } => store_scoped(v, vm.scope(), up, index),
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

fn resolve_arguments<V: Value>(vm: &VirtualMachine<V>, arguments: &[Address]) -> Vec<V> {
    let mut args = vec![];
    for argument in arguments.iter() {
        args.push(load(*argument, vm));
    }
    args
}
