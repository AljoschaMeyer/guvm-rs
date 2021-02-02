use core::future::Future;

use smol::{LocalExecutor, channel::{unbounded, Sender, Receiver}};

use crate::{Continuation, Value};

pub struct EventLoop<V: Value> {
    executor: LocalExecutor<'static>,
    receiver: Receiver<(Result<V, <V as Value>::Failure>, Continuation)>,
    sender: Sender<(Result<V, <V as Value>::Failure>, Continuation)>,
}

impl<V: Value> EventLoop<V> {
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        EventLoop {
            executor: LocalExecutor::new(),
            receiver,
            sender,
        }
    }

    pub(crate) fn spawn<F: 'static + Future<Output = Result<V, <V as Value>::Failure>>>(&mut self, f: F, continuation: Continuation) {
        let sender = self.sender.clone();
        let task = self.executor.spawn(async move {
            let r = f.await;
            sender.send((r, continuation)).await
        });
        task.detach();
    }

    pub(crate) async fn next(&mut self) -> Option<(Result<V, <V as Value>::Failure>, Continuation)> {
        if self.receiver.is_empty() {
            if self.executor.is_empty() {
                return None;
            } else {
                self.executor.tick().await;
            }
        }

        match self.receiver.recv().await {
            Ok(yay) => return Some(yay),
            Err(_) => unreachable!(),
        }
    }
}
