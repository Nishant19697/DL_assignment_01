import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import pandas as pd


(X_train, y_train), (_, _) = fashion_mnist.load_data()

print(f"Dataset shape: {X_train.shape}, Labels shape: {y_train.shape}") 
X_train_flattened = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
y_train_reshaped = y_train.reshape(-1, 1)  
df = pd.DataFrame(X_train_flattened)
df['label'] = y_train_reshaped  
X_train_np = X_train_flattened  
y_train_np = y_train_reshaped.flatten()  


xs = X_train_np.tolist()
ys = y_train_np.tolist()

print(f"Total Samples: {len(xs)}, Unique Labels: {set(ys)}")





class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out


    def __repr__(self):
        return f"Value(name={self.label}, data={self.data}), grad_fn={self._backward}"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            if self.grad.shape[0] != out.grad.shape[0] and isinstance(out.grad, np.ndarray):
              self.grad = self.grad + out.grad.sum(axis=0)
            else:
              self.grad = self.grad + out.grad
            if other.grad.shape[0] != out.grad.shape[0] and isinstance(out.grad, np.ndarray):
              other.grad = other.grad + out.grad.sum(axis=0)
            else:
              other.grad = other.grad + out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
      assert isinstance(other, (int, float))
      out = Value(self.data ** other, (self,), f'**{other}')

      def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    
      out._backward = _backward
      # print(out)
      return out

    def __truediv__(self, other):
      other = other if isinstance(other, Value) else Value(other)
      # print("truediv other : ", other)
      return self * other**-1.0  
    


    def __matmul__(self, other):  
      other = other if isinstance(other, Value) else Value(other)
      # print(self.data.shape, other.data.shape)
      out = Value(self.data @ other.data, (self, other), '@')
      def _backward():
          # print(f"MatMul Backward: out.grad sum = {np.sum(out.grad)}")
          
          if self.grad is None:
              self.grad = np.zeros_like(self.data)
          if other.grad is None:
              other.grad = np.zeros_like(other.data)

          self.grad += out.grad @ other.data.T

          other.grad += self.data.T @ out.grad

      out._backward = _backward
      return out


    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            if self.grad.shape != self.data.shape:
                self.grad = np.zeros_like(self.data)
                
            self.grad += (1 - t**2) * out.grad


        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)  #

        for node in reversed(topo):
            # for prev in node._prev:
            #     print(f"Before {prev.label} {prev._op}: Grad sum = {np.sum(prev.grad)}")
            node._backward()
            # for prev in node._prev:
            #     print(f"Before {prev.label} {prev._op}: Grad sum = {np.sum(prev.grad)}")




class Layer:
    def __init__(self, nin, nout, apply_nonlin=True):
        self.apply_nonlin = apply_nonlin
        bound = np.sqrt(6 / (nin + nout))
        self.w = Value(np.random.uniform(-bound, bound, (nin, nout))) 
        self.b = Value(np.zeros((1, nout))) 
        # print("Linear : ", self.w.data.shape, self.b.data.shape)

    def __call__(self, x):
        # print("Linear input shape : ", x.data.shape, type(x))
        # print("Linear weight shape : ", self.w.data.shape, type(self.w))
        # print("Linear bias shape : ", self.b.data.shape, type(self.b))


        # x = Value(np.array(x).reshape(-1, 1))
        # print("Linear input shape after reshape: ", x.data.shape)

        if self.apply_nonlin:
          out = (x@self.w + self.b).tanh()
          # out.label = "Linear addition"
        else:
          out = x@self.w + self.b
        # print("Linear out : ", out.data.shape)
          # out.label = "Linear addition"
        return out


    def parameters(self):
        return [self.w, self.b]


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts[:-1]
        self.layers = [Layer(sizes[i], sizes[i+1], True) for i in range(len(nouts[:-1]))]
        self.layers.append(Layer(nouts[-2], nouts[-1], False))

    def __call__(self, x):
        # x = np.array(x).reshape(-1, 1)  # Ensure input is column vector
        # print("MLP input shape : ", x.data.shape)
        # print("MLP input range : ", x.data.mean(), x.data.max(), x.data.min())
        for layer in self.layers:
            # print("MLP layer weight range : ", layer.w.data.mean(), layer.w.data.max(), layer.w.data.min())
            # print("MLP layer bias range : ", layer.b.data.mean(), layer.b.data.max(), layer.b.data.min())

            x = layer(x)
            # print("MLP layer out range : ", x.data.mean(), x.data.max(), x.data.min())
            # print("aaaaa : ", x.data.shape)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
#optimizers    

class optimizer:
    def __init__(self, model, learning_rate=0.01, algo="sgd", beta1=0.9, beta2=0.999, eps=1e-8, decay_factor=0.9, momentum=0.9):

        self.model = model
        self.lr = learning_rate
        self.algo = algo.lower()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.decay = decay_factor
        self.momentum = momentum
        self.time_step = 1 
        
        self.vel = {param: np.zeros_like(param.data) for param in model.parameters()}
        self.first_moment = {param: np.zeros_like(param.data) for param in model.parameters()}
        self.second_moment = {param: np.zeros_like(param.data) for param in model.parameters()}
    
    def update(self):
        for param in self.model.parameters():
            if self.algo == "sgd":
                param.data -= self.lr * param.grad 

            elif self.algo == "momentum":
                self.vel[param] = self.momentum * self.vel[param] - self.lr * param.grad
                param.data += self.vel[param]

            elif self.algo == "nesterov":
                prev_velocity = self.vel[param]
                self.vel[param] = self.momentum * self.vel[param] - self.lr * param.grad
                param.data += -self.momentum * prev_velocity + (1 + self.momentum) * self.vel[param]

            elif self.algo == "rmsprop":
                self.second_moment[param] = self.decay * self.second_moment[param] + (1 - self.decay) * (param.grad ** 2)
                param.data -= self.lr * param.grad / (np.sqrt(self.second_moment[param]) + self.eps)

            elif self.algo == "adam":
                self.first_moment[param] = self.beta1 * self.first_moment[param] + (1 - self.beta1) * param.grad
                self.second_moment[param] = self.beta2 * self.second_moment[param] + (1 - self.beta2) * (param.grad ** 2)

                first_moment_corrected = self.first_moment[param] / (1 - self.beta1 ** self.time_step)
                second_moment_corrected = self.second_moment[param] / (1 - self.beta2 ** self.time_step)

                param.data -= self.lr * first_moment_corrected / (np.sqrt(second_moment_corrected) + self.eps)

            elif self.algo == "nadam":
                first_moment_estimate = self.beta1 * self.first_moment[param] + (1 - self.beta1) * param.grad
                self.second_moment[param] = self.beta2 * self.second_moment[param] + (1 - self.beta2) * (param.grad ** 2)

                first_moment_corrected = first_moment_estimate / (1 - self.beta1 ** self.time_step)
                second_moment_corrected = self.second_moment[param] / (1 - self.beta2 ** self.time_step)

                nadam_adjustment = (self.beta1 * first_moment_corrected + (1 - self.beta1) * param.grad) / (np.sqrt(second_moment_corrected) + self.eps)
                param.data -= self.lr * nadam_adjustment

        self.time_step += 1 
    
    def reset_gradients(self):
        for param in self.model.parameters():
            param.grad.fill(0)


class SoftmaxCrossEntropy:
    def __call__(self, x, y):
        batch_size = y.shape[0]
        x.label = "y_pred"
        x_data = x.data 
        # print("x_data shape : ", x_data.shape)
        # max_x = np.max(x_data, axis=0, keepdims=True)  
        # exps = np.exp(x_data - max_x)
        # print(x_data)


        exps = np.exp(x_data)
        # softmax = exps / (1 + exps) #for sigmoid
        softmax = exps / np.sum(exps, axis=1, keepdims=True)   

        # print("softmax shape : ", softmax.shape)
        
        # print(type(y))
        # print(type(softmax))
        # print(y.shape, softmax.shape)
        # Cross-entropy loss

        
        loss_data = -np.log(softmax)  
        mask = np.zeros_like(loss_data)
        mask[np.arange(y.shape[0]), y] = 1
        loss_data = loss_data * mask

        #loss_data = (softmax - y[:, np.newaxis])**2 #for sigmoid

        loss_data = loss_data.sum() / batch_size
        loss = Value(loss_data, (x,), 'softmax_ce')  
        self.softmax = softmax
        self.x = x
        self.y = y

        def _backward():
            batch_size = x.data.shape[0]
            grad = softmax.copy()

            #grad = 2 * (grad - self.y[:, np.newaxis]) * grad * (1 - grad) #for sigmoid

            #full_grad = softmax (1 - softmax)
            
           
        
            y_indices = np.array(y.data, dtype=np.int32)  
            grad[np.arange(batch_size), y_indices] -= 1 

            x.grad += grad 

        loss._backward = _backward
        loss._prev = {x}
        loss.label = "loss"
        return loss, softmax


# n_bits = 4

# xs = [np.binary_repr(i, width=n_bits) for i in np.arange(2**n_bits)]
# xs = np.array([np.array([int(y) for y in x]) for x in xs])

# ys = (xs[:, 0] == 1).astype(int)

# print(xs.shape, ys.shape)

xs = Value(xs)
# ys = Value(ys)

# # print("mean of xs : ", xs.mean(), xs.std())
# xs = np.array(xs)  # Keep xs as Value
ys = np.array(ys)  # Ensure ys remains a NumPy array


model = MLP(784, [512, 256,10])
loss_fn = SoftmaxCrossEntropy()
# lr = 0.01
opt = optimizer(model, learning_rate= 0.01, algo = "momentum")
for k in range(100):
    # print("\n\n", k)
    ypred = model(xs)
    # print(ypred)
    loss, softmax = loss_fn(ypred, ys)
    # for p in model.parameters():
    #     p.grad.fill(0)  

    # loss.backward()

    opt.reset_gradients()
    loss.backward()
    


    print(f"Iteration {k}: Loss = {loss.data}")  
    # grad_norm = []
    # for i, p in enumerate(model.parameters()):
    #     grad_norm.append(np.linalg.norm(p.grad.data))
    # # print("grad norms : ", grad_norm)

    # loss = None


    # for i, p in enumerate(model.parameters()):
    #     # print(p)
    #     # print(f"Layer {i}: Weight sum: {np.sum(p.data)}, Grad sum: {np.sum(p.grad)}")
    #     # p.data -= (lr*0.993**k) * (p.grad + p.data*0.005) 
    #     p.data -= lr*p.grad
    opt.update()

ypred = model(xs)
max_ = np.argmax(ypred.data, axis=1)
correct_predictions = np.sum(max_ == ys)  
total_samples = len(ys)  

accuracy = correct_predictions / total_samples  
accuracy_percentage = accuracy * 100  

print(f"Accuracy: {accuracy:.4f} ({accuracy_percentage:.2f}%)")

print("\n\n")
print("Target : ", ys)
print("Predicted : ", max_)
print(f"Accuracy: {accuracy:.4f} ({accuracy_percentage:.2f}%)")
# print("accuracy:" np.where )
# print("Predicted : ", (ypred.data > 0).astype(int).reshape(-1))
