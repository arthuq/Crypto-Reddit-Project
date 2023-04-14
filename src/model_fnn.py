
## IMPORTATION OF CONFIG FILE

# Selecting dir path for config file
import os
tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"

# Execute config file
exec(open(f'{tmp_path}/configs/config_local.py').read())

# Importing py files
exec(open(f'{SRC_PATH}/csv_import.py').read())

##

device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

##
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

def train_fnn( train_version:list, test_version:list,
                    jump_threshold:float=.06, layer_sizes:list=[8,8,8],
                    n_epochs:int=20, batch_size=8*1024, learning_rate=1e-3,
                    plot_output:bool=False):
    # dates
    indx = get_indx(test_version)

    # Importation of df
    X_train, Y_train = import_several_csv(train_version, to_np = True)
    X_test, Y_test = import_several_csv(test_version, to_np = True)

    # preprocess
    X_train, _ = preprocess_df(X_train, Y_train, jump_threshold, True, norm = NORM)
    X_test, _ = preprocess_df(X_test, Y_test, jump_threshold, True, norm = NORM)

    # to torch
    X = torch.from_numpy(X_train).to(device).float()
    Y = torch.from_numpy(Y_train).to(device).float()
    X_test = torch.from_numpy(X_test).to(device).float()

    # Training of FNN
    trained_phi = jump_by_learning(X, Y, layer_sizes, n_epochs, batch_size, learning_rate)

    # Compute output
    res = trained_phi(X_test).detach().numpy()

    # Rescaling result output
    def rescale(x) : return (x - min(x)) / (max(x) - min(x))

    res = rescale(res)
    Y_test = rescale(Y_test)
    # res *= max(Y_test)

    top = np.mean(Y_test) + np.std(Y_test)
    bot = np.mean(Y_test) - np.std(Y_test)

    # plot
    if plot_output:
        plt.plot(res, "o", label = "Jump proba", markersize=8)

        for i in list(np.where(Y_test > top)[0]) :
            plt.axvspan(i-.5, i+.5, 0.75, 1, facecolor='limegreen', alpha=0.5)

        for i in set(list(np.where(Y_test > bot))[0]) & set(list(np.where(Y_test < top))[0]) :
            plt.axvspan(i-.5, i+.5,0.25, 0.75, facecolor='gold', alpha=0.5)

        for i in list(np.where(Y_test < bot)[0]) :
            plt.axvspan(i-.5, i+.5, 0, 0.25, facecolor='salmon', alpha=0.5)

        plt.title(f"FNN\nVersion {train_version}")
        # plt.xticks(rotation=45); plt.grid()
        plt.legend()
        # plt.show()

    # Saving output
    res = pd.DataFrame(res, index = indx, columns = ["fnn"])
    res.to_csv(f"{RES_PATH}/fnn.csv", index = True)

    return res

## IMPORTING THE TRAINING AND TEST DATA

# Parameters
# train_version = list(range(23, 35))
# test_version = [35, 36]
# jump_threshold = 0.05

layer_sizes = [2,4,8,4,2]

train_fnn(train_version = train_version, test_version=test_version,
                jump_threshold = jump_threshold,layer_sizes = layer_sizes,
                n_epochs= int(1e3), batch_size= 2 * 1024, learning_rate = 1e-5,
                plot_output = True)

plt.savefig(f'{tmp_path}/photos/Forward NN.png')

plt.close()