
## IMPORTATION OF CONFIG FILE

# Selecting dir path for config file
import os
tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"

# Execute config file
exec(open(f'{tmp_path}/configs/config_local.py').read())

# Importing py files
# exec(open(f'{SRC_PATH}/scrapper.py').read())
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

def logic_phi(X, Y, n_epochs, batch_size,
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

def train_logistic( train_version:list, test_version:list,
                    jump_threshold:float=.06,
                    n_epochs:int=20, batch_size=8*1024, learning_rate=1e-3,
                    plot_output:bool=False):

    # dates
    indx = get_indx(test_version)

    # import
    X_train, Y_train = import_several_csv(train_version, to_np = False)
    X_test, Y_test = import_several_csv(test_version, to_np = False)
    Y_test_ = Y_test.copy()
    # preprocess
    X_train, Y_train = preprocess_df(X_train, Y_train, jump_threshold, True, norm = NORM)
    X_test, Y_test = preprocess_df(X_test, Y_test, jump_threshold, True, norm = NORM)

    # to torch
    X_train = torch.from_numpy(X_train).to(device).float()
    Y_train = torch.from_numpy(Y_train).to(device).float()
    X_test = torch.from_numpy(X_test).to(device).float()

    # Training of FNN
    trained_phi = logic_phi(X_train, Y_train, n_epochs, batch_size,
                                            learning_rate=1e-3)

    res = trained_phi(X_test).detach().numpy()
    res = (res - min(res)) / (max(res) - min(res))

    # res *= max(Y_test)




    top = np.mean(Y_test_) + np.std(Y_test_)
    bot = np.mean(Y_test_) - np.std(Y_test_)

    # plot
    if plot_output:
        plt.plot(res, "o", label = "Logistic NN", markersize=8)

        for i in list(np.where(Y_test_ > top)[0]) :
            plt.axvspan(i-.5, i+.5, 0.75, 1, facecolor='limegreen', alpha=0.5)

        for i in set(list(np.where(Y_test_ > bot))[0]) & set(list(np.where(Y_test_ < top))[0]) :
            plt.axvspan(i-.5, i+.5,0.25, 0.75, facecolor='gold', alpha=0.5)

        for i in list(np.where(Y_test_ < bot)[0]) :
            plt.axvspan(i-.5, i+.5, 0, 0.25, facecolor='salmon', alpha=0.5)

        plt.title(f"Logistic NN\nVersion {train_version}")
        # plt.xticks(rotation=45);
        plt.grid()
        plt.legend()
        # plt.show()

    res = pd.DataFrame(res, index = indx, columns = ["logistic"])
    res.to_csv(f"{RES_PATH}/logistic_nn.csv", index = True)


    return res

##
# Parameters
# train_version = list(range(23, 35))
# test_version = [35, 36]
# jump_threshold = 0.05

train_logistic(train_version = train_version, test_version=test_version,
                jump_threshold = jump_threshold,
                n_epochs=300, batch_size=16*1024, learning_rate=1e-6,
                plot_output = True)

plt.savefig(f'{tmp_path}/photos/Logistic NN.png')
plt.close()