import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn


class RecurrentModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RecurrentModel, self).__init__()
        self.s = nn.Parameter(torch.FloatTensor([1e-5, 1, 1]), requires_grad=True)
        self.w = nn.Parameter(torch.FloatTensor(2, 2).uniform_(-1e-5, 1e-5), requires_grad=True)
        self.p = nn.Parameter(torch.FloatTensor(2).uniform_(-1e-5, 1e-5), requires_grad=True)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = 0.1
        Wc = recurrent_W(3, self.s, self.w, self.p, h)
        y = x.mm(Wc).sum(1)
        return y

x = torch.randn(10, requires_grad=True)
y = x * 2
z = y.mean()

print(z.requires_grad)

z.backward()

print(x.grad)

def recurrent_kernel(u, v, s, w, p, h):
    uv = Variable(torch.FloatTensor([u, v]))
    return s[0] + w.mv(uv).sub_(p).cos().dot(s[1:]) * h

def recurrent_integrate(fun, a, b, N=100):
    res = 0
    h = (b - a) / N

    for i in np.linspace(a, b, N):
        res += fun(a + i, h) * h

    return res

def recurrent_V(v, n, s, w, p, h):
    fun = lambda u, h: recurrent_kernel(u, v, s, w, p, h).mul_(u - n)
    return recurrent_integrate(fun, n, n+1)

def recurrent_Q(v, n, s, w, p, h):
    fun = lambda u, h: recurrent_kernel(u, v, s, w, p, h)
    return recurrent_integrate(fun, n, n+1)

def recurrent_W(N, s, w, p, h):
    Qp = lambda v, n: recurrent_Q(v, n, s, w, p, h)
    Vp = lambda v, n: recurrent_V(v, n, s, w, p, h)

    W = torch.zeros((N, N))

    W[0, :N-1] = torch.cat([ (Qp(v, 1) - Vp(v, 1)).unsqueeze(0) for v in range(1, N)])
    for j in range(2, N):
        W[j-1, :N-j] = torch.cat([ (Qp(v, j) - Vp(v, j) + Vp(v, j - 1)).unsqueeze(0) for v in range(1, N-j+1)])
    W[N-1, :N-1] = torch.cat([ Vp(v, N-1).unsqueeze(0) for v in range(1, N)])

    return W.t()


s = Variable(torch.FloatTensor([1e-5, 1, 1]), requires_grad=True)
w = Variable(torch.FloatTensor(2, 2).uniform_(-1e-5, 1e-5), requires_grad=True)
p = Variable(torch.FloatTensor(2).uniform_(-1e-5, 1e-5), requires_grad=True)

optimizer = torch.optim.Adam([s, w, p], lr=1e-3)

data_x_t = torch.FloatTensor(100, 3).uniform_()
data_y_t = data_x_t.mm(torch.FloatTensor([[1, 2, 3]]).t_()).view(-1)


alpha = -1e-3

train_size = int(0.8 * len(data_x_t))
train_x_t = data_x_t[:train_size]
train_y_t = data_y_t[:train_size]
test_x_t = data_x_t[train_size:]
test_y_t = data_y_t[train_size:]

l1_lambda = 1e-5
l2_lambda = 1e-4

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    ])

train_x_augmented = []
train_y_augmented = []
for i in range(len(train_x_t)):
    augmented_data = data_transform(train_x_t.unsqueeze(0)).squeeze(0)
    train_x_augmented.append(augmented_data.squeeze(0))
    train_y_augmented.append(train_y_t[i])

train_x_t = torch.stack(train_x_augmented)
train_y_t = torch.tensor(train_y_augmented)

patience = 10
best_loss = float('inf')
epochs_without_improvement = 0


for i in range(100):
    data_x, data_y = Variable(data_x_t), Variable(data_y_t)
    train_x, train_y = Variable(train_x_t), Variable(train_y_t)

    h = 0.1
    Wc = recurrent_W(3, s, w, p, h)
    y = data_x.mm(Wc).sum(1)
    loss = data_y.sub(y).pow(2).mean()

    l1_regularization = l1_lambda * (torch.abs(s).sum() + torch.abs(w).sum() + torch.abs(p).sum())
    loss += l1_regularization

    l2_regularization = l2_lambda * (torch.square(s).sum() + torch.square(w).sum() + torch.square(p).sum())
    loss += l2_regularization

    optimizer.zero_grad()
    optimizer.step()

    print(loss.data.item())

    if loss.data.item() < best_loss:
        best_loss = loss.data.item()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stop on the era: {i}")
        break

    loss.backward()
    s.data.add_(s.grad.data.mul(alpha))
    s.grad.data.zero_()

    w.data.add_(w.grad.data.mul(alpha))
    w.grad.data.zero_()

    p.data.add_(p.grad.data.mul(alpha))
    p.grad.data.zero_()

    test_x, test_y = Variable(test_x_t), Variable(test_y_t)
    Wc = recurrent_W(3, s, w, p, h)
    y_pred = test_x.mm(Wc).sum(1)
    test_loss = test_y.sub(y_pred).pow(2).mean()

    print("Test loss:", test_loss.data.item(), i)
