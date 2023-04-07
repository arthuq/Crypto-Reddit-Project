
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

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)

     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs

def phi_by_learning_logistic(X, Y, n_epochs, batch_size,
                    learning_rate=1e-3, device=device):
    data_size, input_size = X.shape
    _, output_size = Y.shape
    n_update = data_size // batch_size

    # Settging up model
    phi = LogisticRegression(input_size, output_size).to(device)
    # optimizer = torch.optim.Adam(phi.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(phi.parameters(), lr = learning_rate)
    criterion = torch.nn.BCELoss()


    for n in range(n_epochs):
        indexes = torch.randperm(data_size-1)
        for k in range(n_update):
            idx = indexes[k*batch_size:(k+1)*batch_size]
            loss = criterion(phi(X[idx]), Y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return phi


## DATA

def train_logistic(versions:int=[VERSION], jump_threshold:float=0.5,
                    n_epochs:int=20, batch_size=8*1024, learning_rate=1e-3,
                    plot_output:bool=False):

    # Importation of df
    x, y = import_several_csv(versions, jump_threshold)
    X = torch.from_numpy(x).to(device).float()
    Y = torch.from_numpy(y).to(device).float()

    # Parameters
    # _, input_size = X.shape
    # _, output_size = Y.shape
    # phi = NeuralNetwork(input_size, layer_sizes, output_size).to(device)

    # Checking for Nan
    if X.isnan().int().sum() + Y.isnan().int().sum() > 0 :
        print("Found nan in X or Y")
        return 0

    # Training of FNN
    trained_phi = phi_by_learning_logistic(X, Y, n_epochs, batch_size,
                                            learning_rate=1e-3)
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
jump_threshold = 0.075

train_logistic(versions = train_version,
                jump_threshold = jump_threshold,
                n_epochs=30, batch_size=16*1024, learning_rate=1e-4,
                plot_output = True)

