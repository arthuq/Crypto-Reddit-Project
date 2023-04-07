
## IMPORTATION OF CONFIG FILE

# Selecting dir path for config file
import os
tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"

# Execute config file
exec(open(f'{tmp_path}/configs/config_local.py').read())

# Importing py files
exec(open(f'{SRC_PATH}/scrapper.py').read())
exec(open(f'{SRC_PATH}/csv_import.py').read())


##

device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

##
"""
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        layers = [ nn.Linear(input_size, layer_sizes[0]), nn.ReLU() ]
        for (ls_in, ls_out) in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(ls_in, ls_out))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        layers.append(nn.Sigmoid())
        self.linear_sigmoid_stack = nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear_sigmoid_stack(x)
        return out
"""
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        layers = [ nn.Linear(input_size, layer_sizes[0]),
                   nn.ReLU() ]
        for (ls_in, ls_out) in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(ls_in, ls_out))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


def jump_by_learning(X, Y, layer_sizes, n_epochs, batch_size, learning_rate=1e-3, device=device):
    data_size, input_size = X.shape
    _, output_size = Y.shape
    n_update = data_size // batch_size

    phi = NeuralNetwork(input_size, layer_sizes, output_size).to(device)
    optimizer = torch.optim.Adam(phi.parameters(), lr=learning_rate)

    for n in range(n_epochs):
        indexes = torch.randperm(data_size-1)
        for k in range(n_update):
            idx = indexes[k*batch_size:(k+1)*batch_size]
            loss = ((Y[idx] - phi(X[idx]))**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return phi


## DATA

def train_FNN(versions:int=[VERSION], jump_threshold:float=0.5, layer_sizes:list=[8,16,16,8], plot_output:bool=False):
    # Importation of df
    x, y = import_several_csv(versions, jump_threshold)
    X = torch.from_numpy(x).to(device).float()
    Y = torch.from_numpy(y).to(device).float()
    # _, input_size = X.shape
    # _, output_size = Y.shape

    # Checking for Nan
    if X.isnan().int().sum() + Y.isnan().int().sum() > 0 :
        print("Found nan in X or Y")
        return 0

    # Training of FNN
    # phi = NeuralNetwork(input_size, layer_sizes, output_size).to(device)
    trained_phi = jump_by_learning(X, Y, layer_sizes, 20, 2*8*1024, learning_rate=1e-4)
    res = trained_phi(X).detach().numpy()
    if plot_output:
        plt.plot(res, "-o", label = "train")
        plt.plot(Y, "*", label = "true")
        plt.title(f"Version {versions}")
        plt.legend(); plt.grid(); plt.show()
    return res

##
# Parameters
train_version = list(range(23, 35))
jump_threshold = 0.1
layer_sizes = [8, 16, 8]

train_FNN(versions = train_version,
            jump_threshold = jump_threshold,
            layer_sizes = layer_sizes,
            plot_output = True)

