
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float32)  # Ensure NumPy array
        self.grad = np.zeros_like(self.data, dtype=np.float32)  # Initialize gradient as array
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __mul__(self, other):
        # print("mul other : ", other.data.shape)
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # print("mul grad shapes : ", self.grad.data.shape, other.grad.data.shape, out.grad.data.shape)
            self.grad += other.data * out.grad  # Element-wise multiplication
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
      assert isinstance(other, (int, float)), "Only int/float powers are supported"
      out = Value(self.data ** other, (self,), f'**{other}')

      def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    
      out._backward = _backward
      # print(out)
      return out

    def __truediv__(self, other):
      other = other if isinstance(other, Value) else Value(other)
      # print("truediv other : ", other)
      return self * other**-1.0  # Uses the power function for division
    


    def __matmul__(self, other):  # Matrix Multiplication
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
        self.grad = np.ones_like(self.data)  # Start gradient with ones (for scalar loss)

        # print("\n===== BACKWARD =====")
        for node in reversed(topo):
            # for prev in node._prev:
            #     print(f"Before {prev.label} {prev._op}: Grad sum = {np.sum(prev.grad)}")
            node._backward()
            # for prev in node._prev:
            #     print(f"Before {prev.label} {prev._op}: Grad sum = {np.sum(prev.grad)}")




class Layer:
    def __init__(self, nin, nout, apply_nonlin=True):
        self.apply_nonlin = apply_nonlin
        self.w = Value(np.random.uniform(-1, 1, (nin, nout)))  # (nout, nin)
        self.b = Value(np.random.uniform(-1, 1, (1, nout)))  # (nout, 1)
        # print("Linear : ", self.w.data.shape, self.b.data.shape)

    def __call__(self, x):
    #     print("Linear input shape : ", x.data.shape, type(x))
    #     print("Linear weight shape : ", self.w.data.shape, type(self.w))
    #     print("Linear bias shape : ", self.b.data.shape, type(self.b))


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
        for layer in self.layers:
            x = layer(x)
            # print("aaaaa : ", x.data.shape)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

class SoftmaxCrossEntropy:
    def __call__(self, x, y):
        x.label = "y_pred"
        x_data = x.data  # Extract NumPy array from Value
        # print("x_data shape : ", x_data.shape)
        max_x = np.max(x_data, axis=0, keepdims=True)  # Numerical stability
        exps = np.exp(x_data - max_x)
        softmax = exps / np.sum(exps, axis=0, keepdims=True)  # Softmax probabilities
        
        # Cross-entropy loss
        loss_data = -np.log(softmax[y])  # Scalar loss
        loss_data = loss_data.sum()/loss_data.shape[0]
        loss = Value(loss_data, (x,), 'softmax_ce')  # Track computation graph

        # Save softmax probabilities for backward pass
        self.softmax = softmax
        self.x = x
        self.y = y

        def _backward():
            batch_size = x.data.shape[0]
            grad = softmax.copy()

            # ✅ Convert y.data into a proper NumPy array for indexing
            y_indices = np.array(y.data, dtype=np.int32)  
            grad[np.arange(batch_size), y_indices] -= 1  # Correct one-hot update
            grad /= batch_size  # Normalize gradient

            x.grad += grad  # Accumulate gradients properly

        loss._backward = _backward
        loss._prev = {x}
        loss.label = "loss"
        return loss, softmax

# # print(xs)
# xs = np.array(xs)
# xs /= 255
# xs -= 0.5
# xs *= 2

# print("mean of xs : ", xs.mean(), xs.std())
xs = np.array(xs)  # Keep xs as Value
ys = np.array(ys)  # Ensure ys remains a NumPy array


model = MLP(784, [256, 128, 2])
loss_fn = SoftmaxCrossEntropy()
lr = 0.01

for k in range(1000):
    # print("\n\n", k)
    ypred = model(xs)
    loss, softmax = loss_fn(ypred, ys)

    # Reset gradients to zero
    for p in model.parameters():
        p.grad.fill(0)  

    loss.backward()

    # if k%100 == 0:
    print(f"Iteration {k}: Loss = {loss.data}")  # Check if loss is decreasing
    grad_norm = []
    for i, p in enumerate(model.parameters()):
      grad_norm.append(np.linalg.norm(p.grad.data))
    print("grad norms : ", grad_norm)

    
    for i, p in enumerate(model.parameters()):
        # print(p)
        # print(f"Layer {i}: Weight sum: {np.sum(p.data)}, Grad sum: {np.sum(p.grad)}")
        p.data -= (lr*0.993**k) * (p.grad + p.data*0.005)  # ✅ Weight update






