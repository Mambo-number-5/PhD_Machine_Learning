from readfolders import KeyErrorMessage, CustomDataset, generate_clean_dataset
from sklearn.model_selection import train_test_split
from os.path import join, isfile
from os import makedirs
import torch
import pandas as pd
from torch import nn
from pprint import pprint
from scipy import stats
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# back_end_matplotlib = 'Qt5Agg'
# use(back_end_matplotlib)
checkpoints_folder = join(".", "Autoencoder_checkpoints")


def get_custom_dataset_one_class_per_defect(from_date: str = "", time_series: bool = None,
                                            build_interactive_filter: bool = False, filter_defects: dict or None = None,
                                            merge_chem_on_iba: bool = False, clean: bool = True,
                                            normalize: bool = True) -> (CustomDataset, CustomDataset, str):
    x, y = generate_clean_dataset(from_date=from_date, timeseries=time_series,
                                  build_interactive_filt=build_interactive_filter, filter_defs=filter_defects,
                                  merge_chem_on_iba=merge_chem_on_iba, clean=clean)
    for i in y.columns:
        x_train, x_test, y_train, y_test = train_test_split(x, y[i], test_size=0.2, random_state=0, stratify=y[i])
        if normalize:
            means = x_train.mean()
            stds = x_train.std()
            x_train = (x_train - means) / stds
            x_test = (x_test - means) / stds
            del means, stds
        to_remove = y_train[y_train == 1].index
        x_prov = x_train.drop(to_remove)
        x_tes_plus = pd.concat([x_test, x_train.loc[to_remove]])
        y_tes_plus = pd.concat([y_test, y_train.loc[to_remove]])
        dat_prov = CustomDataset(data=x_prov)
        dat_test = CustomDataset(data=x_tes_plus, labels=y_tes_plus)
        del to_remove, x_train, y_train, x_test, y_test, x_tes_plus, y_tes_plus, x_prov
        yield dat_prov, dat_test, i


def error_messages(name: str, more_specific: str = "", *args) -> str:
    if name == "instructions":
        instructions = "For building the auto-encoder should be given layer_1, layer_2, ..., layer_N parameters to "
        instructions += "generate a symmetric auto-encoder.\n To manually generate the layers of the auto-encoder "
        instructions += "should be given parameters \"enc_layer_1, enc_layer_2, ..., enc_layer_N\" for the encoder part"
        instructions += " and \"dec_layer_M, dec_layer_(M-1), ..., dec_layer_1, dec_layer_0\" for the decoder part.\n"
        instructions += "In case of symmetric auto-encoder the links between layer will be:\n"
        instructions += "(enc_)layer_1 -> (enc_)layer_2 -> ... -> (enc_)layer_N -> (dec_)layer_[N-1] -> "
        instructions += "(dec_)layer_[N-2] -> ... -> (dec_)layer_1 -> (dec_)layer_0\n "
        instructions += "In case of manual auto-encoder the links between the layers will be:\n"
        instructions += "enc_layer_1 -> enc_layer_2 -> ... -> enc_layer_N -> dec_layer_M -> dec_layer_[M-1] -> "
        instructions += "dec_layer_[M-2] -> ... -> dec_layer_1 -> dec_layer_0\n"
        if more_specific == "key_disambiguate":
            msg = "Parameters for symmetrical and manual auto-encoder have been found. Please pick only one type\n"
            return msg + instructions
        elif more_specific == "only_decoder":
            msg = "Layers only for the decoder part of the auto-encoder have been found, "
            msg += "it will be necessary to have at least one layer in the encoder part.\n"
            return msg + instructions
        elif more_specific == "no_layers":
            msg = "Layers parameters have not been found.\n"
            return msg + instructions

    elif name == "only_encoder":
        msg = "Layers only for the encoder part of the manually generated auto-encoder have been found, "
        msg += "there will be only the layer_0 in the decoder part"
        return msg

    elif name == "sym_key_name":
        build_msg = "In the building of the symmetric auto-encoder you should start with layer_1 and give a "
        build_msg += "progressive number to the layers: layer_1, layer_2, layer_3, ..., layer_N"

        if more_specific == "layer_1":
            msg = "During the construction of the symmetric auto-encoder, starting layer_1 not found:\n"
            return msg + build_msg

        elif more_specific == "layer_expected":
            msg = "Error during the construction of the symmetric auto-encoder, layer_{} expected but {}".format(*args)
            msg += " was found.\n"
            return msg + build_msg

        elif more_specific == "layer_name":
            msg = "Error in the name of the layer: '{}' found, while 'layer_{}' expected".format(*args)
            return msg + build_msg

    elif name == "sym_out_of_threshold":
        msg = "The symmetrical neural network should have layers that code and decode information, so consecutive "
        msg += "layers should have progressively less neurons:\n"
        msg += "{} has {} neurons while {} has {} neurons.".format(*args)
        return msg

    elif name == "layer_value_not_int":
        msg = "The value of "
        msg += "{} should be an int that represent the number of neurons for that layer, '{}' found ".format(*args)
        msg += "instead.\n"
        return msg

    raise KeyError("Error during the use of the key for the choice of the error message")


class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        layers_sym = sorted([k for k in kwargs.keys() if k.startswith('layer_')])
        layers_enc = sorted([k for k in kwargs.keys() if k.startswith('enc_layer_')])
        layers_dec = sorted([k for k in kwargs.keys() if k.startswith('dec_layer_')], reverse=True)
        s = bool(layers_sym)
        e = bool(layers_enc)
        d = bool(layers_dec)

        if s and (e or d):
            raise KeyError(KeyErrorMessage(error_messages("instructions", "key_disambiguate")))
        elif e and not d:
            raise Warning(error_messages("only_encoder"))
        elif not s and not e:
            if d:
                raise KeyError(KeyErrorMessage(error_messages("instructions", "only_decoder")))
            else:
                raise KeyError(KeyErrorMessage(error_messages("instructions", "no_layers")))

        try:
            before = kwargs['input_shape']
        except KeyError:
            raise KeyError("No input shape specified, insert a input_shape parameter")

        self.activation = kwargs.get("activation_function", torch.relu)

        f = kwargs.get("final_activation")
        if f is not None:
            if isinstance(f, bool):
                self.final_activation = f
            else:
                raise KeyError("The final_activation parameter should be a boolean")
        else:
            self.final_activation = False
        del d, e, f

        self.is_symmetric = s

        if s:
            del s, layers_enc, layers_dec

            if layers_sym[0] != "layer_1":
                raise KeyError(KeyErrorMessage(error_messages("sym_key_name", "layer_1")))

            max_thresh = float('inf')
            lay_bef = None
            neurons = list()
            for i, l in enumerate(layers_sym, start=1):
                try:
                    num_layer = int(l[6:])
                    if num_layer != i:
                        raise KeyError(KeyErrorMessage(error_messages("sym_key_name", "layer_expected", i, l)))
                except ValueError:
                    raise KeyError(KeyErrorMessage(error_messages("sym_key_name", "layer_name", l, i)))

                try:
                    val = int(kwargs[l])
                except ValueError:
                    raise ValueError(error_messages("layer_value_not_int", "", l, kwargs[l]))
                else:
                    if val > max_thresh:
                        raise ValueError(error_messages("sym_out_of_threshold", "", lay_bef, max_thresh, l, val))
                    else:
                        neurons.append(val)
                        max_thresh = val
                        lay_bef = l
            del i, l, val, max_thresh, num_layer, lay_bef

            in_out = list()
            lvl = "encoding_layer_"
            # Encoding part networks
            for i, neu in enumerate(neurons, start=1):
                name_layer = lvl + str(i)
                setattr(self, name_layer, nn.Linear(in_features=before, out_features=neu))
                before = neu
                in_out.append(name_layer)

            num_lev = len(neurons) - 1
            lvl = "decoding_layer_"

            # Decoding part networks
            for lev in neurons[-2::-1]:
                name_layer = lvl + str(num_lev)
                in_out.append(name_layer)
                setattr(self, name_layer, nn.Linear(in_features=before, out_features=lev))
                before = lev
                num_lev -= 1
            name_layer = lvl + str(num_lev)
            in_out.append(name_layer)
            setattr(self, name_layer, nn.Linear(in_features=before, out_features=kwargs['input_shape']))

            self.layers = ('Symmetric:', tuple(neurons))
            self.layers_flow = tuple(in_out)
            self.name = "Autoencoder_S_" + "_".join((str(i) for i in neurons)) + "_"

        else:
            del s, layers_sym
            # TO DO !!!

    def forward(self, features: torch.Tensor):
        response = features.detach().clone()
        for layers in self.layers_flow[:-1]:
            response = getattr(self, layers)(response)
            response = self.activation(response)
        out = getattr(self, self.layers_flow[-1])(response)
        if self.final_activation:
            out = self.activation(out)
        return out


def train_defect(model_dict: dict, data: CustomDataset, batch_size: int, epochs: int = 5, save: str = "") -> AE:
    train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=8)
    if isinstance(model_dict, dict):
        model_dict['input_shape'] = data.data.shape[1]
        model = AE(**model_dict).to(device)
    else:
        raise KeyError("model given is not a dictionary")

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())
    loss = None
    total_step = len(train_loader)
    # The train for loop
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            batch = batch.float().to(device)
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch)

            # optimizer
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (i + 1) % 10 == 0:
                print("Epoch [{}/{}], batch [{}/{}], Loss: {:.4f}".format(epoch + 1, epochs, i + 1, total_step,
                                                                          loss.item()))
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, epochs, loss.item()))

    if save != "":
        save_str = join(checkpoints_folder, model.name + save + ".ckpt")
        torch.save(model.state_dict(), save_str)
    return model


def train_every_defect(model_dict: dict, from_date: str = "", time_series: bool = None,
                       build_interactive_filter: bool = False, filter_defects: dict or None = None,
                       merge_chem_on_iba: bool = False, clean: bool = True, normalize: bool = True,
                       batch_size: int = 7_500, epoch_dict: dict or None = None, save_model: bool = True) -> dict:
    result = dict()
    if epoch_dict is None:
        epoch_dict = {}
    generator_dfs = get_custom_dataset_one_class_per_defect(from_date=from_date, time_series=time_series,
                                                            build_interactive_filter=build_interactive_filter,
                                                            filter_defects=filter_defects,
                                                            merge_chem_on_iba=merge_chem_on_iba, clean=clean,
                                                            normalize=normalize)
    print('Autoencoder:')
    pprint(model_dict)
    print("going to be trained for each defect...")
    for custom_tr, custom_te, defect_str in generator_dfs:
        print(f"Defect {defect_str} in training:")
        eps = epoch_dict.get(defect_str, 10)
        if save_model:
            model = train_defect(model_dict, custom_tr, batch_size, epochs=eps, save=defect_str)
        else:
            model = train_defect(model_dict, custom_tr, batch_size, epochs=eps)
        result[defect_str] = model
        print('\n')
    return result


def get_dict_from_name(name: str) -> dict:
    err_str = f"{name} do not seem the name of an autoencoder model saved"
    if name.startswith("Autoencoder_") and name.endswith(".ckpt"):
        name = name[:-5]
    else:
        raise ValueError(err_str)
    bound_up = -1  # Defect in name
    name_list = name.split("_")
    type_ae = name_list[1]
    j = None
    if type_ae == "S":
        out_dict = dict()
        try:
            for i, j in enumerate(name_list[2: bound_up], start=1):
                name_layer = 'layer_' + str(i)
                out_dict[name_layer] = int(j)
            return out_dict
        except ValueError:
            raise ValueError(err_str)
    elif type_ae == "E":
        out_dict = dict()
        i = None
        try:
            for i, j in enumerate(name_list[2: bound_up], start=1):
                name_layer = "enc_layer_" + str(i)
                out_dict[name_layer] = int(j)
        except ValueError:
            if j != "D":
                raise ValueError(err_str)
            i += 2  # To begin after the D
        name_dec_layer = i - 1
        try:
            for j in name_list[i: bound_up]:
                name_layer = "dec_layer_" + str(name_dec_layer)
                out_dict[name_layer] = int(j)
            return out_dict
        except ValueError:
            raise ValueError(err_str)
    else:
        raise ValueError(err_str)


def gaussian_autoencoder(model: str or AE,  custom_train_dataset: CustomDataset, custom_test_dataset: CustomDataset,
                         model_for_defect: str = "", btc_size: int = 7_500, figure_name: str = ""):
    train_loader = torch.utils.data.DataLoader(dataset=custom_train_dataset, batch_size=btc_size, shuffle=True,
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=custom_test_dataset, batch_size=btc_size, shuffle=True,
                                              num_workers=8)
    if isinstance(model, str):
        weights_model = torch.load(join(checkpoints_folder, model))
        model_dict = get_dict_from_name(model)
        model_dict['input_shape'] = custom_train_dataset.data.shape[1]
        model = AE(**model_dict)
        model.load_state_dict(weights_model)
        del weights_model, model_dict
    elif not isinstance(model, AE):
        raise KeyError("Problem with the type of the model parameter")
    else:
        model.to('cpu')

    with torch.no_grad():
        result_train = list()
        for batch in train_loader:
            batch = batch.float()
            reconstructed = model(batch)
            losses = ((reconstructed - batch) ** 2).mean(dim=1)
            result_train.append(losses)
        total = torch.cat(result_train)
        n = len(total)
        mean_ml = total.mean()
        var_ml = n / (n - 1) * total.var(dim=0)  # Give the unbiased (due to Maximum Likelihood) value of the variance
        std_ml = torch.sqrt(var_ml)
        del total, result_train, losses, n, reconstructed, batch, train_loader, var_ml

        result_test = list()
        for batch, labels in test_loader:
            batch = batch.float()
            reconstructed = model(batch)
            losses = ((reconstructed - batch) ** 2).mean(dim=1)
            df_losses = pd.DataFrame(losses.numpy(), columns=['Losses'])
            df_losses['Anomaly'] = labels.numpy()
            result_test.append(df_losses)
        df_test = pd.concat(result_test, ignore_index=True)
        class_0 = df_test[df_test['Anomaly'] == 0]['Losses']
        class_1 = df_test[df_test['Anomaly'] == 1]['Losses']
        del losses, reconstructed, batch, labels, df_losses, test_loader, result_test, df_test

    plot_normal_curve(mean_ml, std_ml, class_0, class_1, model, defect_str=model_for_defect, filename=figure_name)


def train_test_one_class_autoencoder(to_test: str, batch_size: int = 7_500, train: bool = None,
                                     number_epoch_train: int = 5, save_train: bool = True, test: bool = False,
                                     figure_test_name: str = "", from_date: str = "", time_series: bool = None,
                                     build_interactive_filter: bool = False, filter_defects: dict or None = None,
                                     merge_chem_on_iba: bool = False, clean: bool = True, normalize: bool = True):
    if not train and not test:
        raise KeyError("This function should be used to train or test an autoencoder")
    generator_one_class = get_custom_dataset_one_class_per_defect(from_date=from_date, time_series=time_series,
                                                                  build_interactive_filter=build_interactive_filter,
                                                                  filter_defects=filter_defects,
                                                                  merge_chem_on_iba=merge_chem_on_iba, clean=clean,
                                                                  normalize=normalize)
    for custom_train_one, custom_test_one, defect_string in generator_one_class:
        if train:
            model_dict = get_dict_from_name(to_test)
            if save_train:
                model = train_defect(model_dict=model_dict, data=custom_train_one, batch_size=batch_size,
                                     epochs=number_epoch_train, save=defect_string)
            else:
                model = train_defect(model_dict=model_dict, data=custom_train_one, batch_size=batch_size,
                                     epochs=number_epoch_train)
            del model_dict
        elif train is None:
            model = to_test + defect_string + ".ckpt"
            if not isfile(join(checkpoints_folder, model)):
                model_dict = get_dict_from_name(model)
                msg = f"Model checkpoint {model} not found starting training"
                if save_train:
                    print(msg + " and saving the checkpoint...")
                    model = train_defect(model_dict=model_dict, data=custom_train_one, batch_size=batch_size,
                                         epochs=number_epoch_train, save=defect_string)
                else:
                    print(msg + "...")
                    model = train_defect(model_dict=model_dict, data=custom_train_one, batch_size=batch_size,
                                         epochs=number_epoch_train)
                del msg, model_dict
            else:
                msg = f"Model checkpoint {model} found, skipping training..."
                print(msg)
                del msg
        else:
            model = to_test + defect_string + ".ckpt"
        
        if test:
            if figure_test_name == "auto":
                if isinstance(model, str):  # Give automatically a name to the figure
                    fold_path = join(".", "Autoencoder results", to_test[:-1])
                    figure_name = join(fold_path, model[:-5])
                    makedirs(fold_path, exist_ok=True)
                else:
                    fold_path = join(".", "Autoencoder results", to_test[:-1])
                    figure_name = join(fold_path, model.name + defect_string)
                    makedirs(fold_path, exist_ok=True)
            else:
                figure_name = figure_test_name
            gaussian_autoencoder(model=model, custom_train_dataset=custom_train_one,
                                 custom_test_dataset=custom_test_one, model_for_defect=defect_string,
                                 btc_size=batch_size, figure_name=figure_name)


def score_autoencoder(gaussian_mean: torch.float32, gaussian_std: torch.float32, class_0: pd.Series,
                      class_1: pd.Series) -> pd.DataFrame:

    sigmas_for_threshold = (1, 2, 3)
    column_titles = ("True Positive", "False Positive", "False Negative", "True Negative", "F1-score")
    result_dataframe = pd.DataFrame(columns=column_titles)
    del column_titles
    for i in sigmas_for_threshold:
        index_for_threshold = str(i) + "σ"
        threshold = gaussian_mean + i * gaussian_std
        threshold = threshold.numpy()
        tp = sum(class_1 > threshold)
        fn = sum(class_1 < threshold)
        tn = sum(class_0 < threshold)
        fp = sum(class_0 > threshold)
        f1 = 2 * tp / (2 * tp + fn + fp)
        values = [int(i) for i in (tp, fp, fn, tn)] + [f1]
        result_dataframe.loc[index_for_threshold] = values
    result_dataframe.index.name = "Threshold"
    return result_dataframe


def plot_normal_curve(mean: torch.float32, std_dev: torch.float32, not_defected: pd.Series, defected: pd.Series,
                      model: AE, defect_str: str, filename: str = ""):

    title = model.name + " for defect " + defect_str
    legend_0 = f"NOT {defect_str}"
    legend_1 = defect_str

    x = np.linspace(mean - std_dev, max(not_defected.max(), defected.max(), mean + 3 * std_dev), 100_000)
    y = stats.norm.pdf(x, mean, std_dev)
    max_y = np.max(y)

    fig = make_subplots(rows=1, cols=1)

    # Plot della curva normal
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                             name=f'Gaussian curve (Mean: {mean}, Standard deviation: {std_dev})'))

    # Scatter plot della prima series
    y_not_defected = stats.norm.cdf(not_defected, mean, std_dev) * max_y
    fig.add_trace(go.Scatter(x=not_defected, y=y_not_defected, mode='markers', name=legend_0,
                             marker=dict(color='blue', symbol='circle', size=8)))

    # Scatter plot della second series
    y_defected = stats.norm.cdf(defected, mean, std_dev) * max_y
    fig.add_trace(go.Scatter(x=defected, y=y_defected, mode='markers', name=legend_1,
                             marker=dict(color='red', symbol='x', size=10)))

    # Impost del grafico
    fig.update_layout(title=title, xaxis_title='Losses', yaxis_title='Y', showlegend=True)

    # Show the interactive figure or save it as HTML
    if filename != "":
        fig.write_html(filename + '.html')
        df_score = score_autoencoder(mean, std_dev, class_0=not_defected, class_1=defected)
        gfg = df_score.to_markdown(index=True, tablefmt="grid")
        with open(filename + ".txt", "w", encoding="utf-8") as f:
            f.write(f'Gaussian curve (Mean: {mean}, Standard deviation: {std_dev})\n')
            print(gfg, file=f)
    else:
        fig.show()


def main(list_of_autoencoder_name: [str], train_: str = "checkpoint", figure_save: bool = True, batch_size: int = 7_500,
         number_epoch_train: int = 5, save_train: bool = True, test: bool = True, from_date: str = "",
         time_series: bool = None, build_interactive_filter: bool = False, filter_defects: dict or None = None,
         merge_chem_on_iba: bool = False, clean: bool = True, normalize: bool = True):
    # Set up of the parameter for the function train_test_one_class_autoencoder
    if train_ == "checkpoint" or train_ == "Checkpoint" or train_ == "C":
        train = None
    elif train_ == "always" or train_ == "Always" or train_ == "A":
        train = True
    elif train_ == "never" or train_ == "Never" or train_ == "N":
        train = False
    else:
        err_msg = "The train parameter can only be only one of this optinons:\n'checkpoint' to train the model only if "
        err_msg += "there is not checkpoint saved in the working directory\n'always' to always train the model\n'never'"
        err_msg += " to never train the model (if there is no checkpoint file an exception will be raised)"
        raise KeyError(KeyErrorMessage(err_msg))
    if figure_save:
        figure_test_name = "auto"
    else:
        figure_test_name = ""
    del train_, figure_save
    # TO TEST FUNCTION ABOVE
    for autoencoder in list_of_autoencoder_name:
        train_test_one_class_autoencoder(to_test=autoencoder, batch_size=batch_size, train=train,
                                         number_epoch_train=number_epoch_train, save_train=save_train, test=test,
                                         figure_test_name=figure_test_name, from_date=from_date,
                                         time_series=time_series, build_interactive_filter=build_interactive_filter,
                                         filter_defects=filter_defects, merge_chem_on_iba=merge_chem_on_iba,
                                         clean=clean, normalize=normalize)


if __name__ == '__main__':
    levels_test = [(80, 50, 30), (128, 64), (140, 120, 100, 80), (120, 100, 80, 50), (128, 80, 64)]
    str_levels = list()
    for tup in levels_test:
        lev = [str(t) for t in tup]
        str_levels.append("_".join(lev))
    autoencoder_names = ["Autoencoder_S_" + i + "_" for i in str_levels]
    main(autoencoder_names)
