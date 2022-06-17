import json
import pickle
import re
import torch 
from torch import Tensor
from zmq import device
def load_json(path):
    """
    Load json dataset into python object (dictionary).

    Args:
        path (str): file path

    Returns:
        Dict: The data object
    """
    with open(path, 'r', encoding='utf-8') as fr:
        data_json = json.load(fr)
    fr.close()
    return data_json

def save_json(obj, path):
    with open(path, 'w') as outfile:
        json.dump(obj, outfile, indent=2)

def save_parameter(save_object, save_file):
    """
    Write the pickled representation of the object to file.

    Args:
        save_object (Any): can be list, dataframe, tensor, nd array, or any type
        save_file (str): file path

    Returns:
        None
    Notes:
        if save_file does not exist, it will be created automatically.
    """
    with open(save_file, 'wb') as f:
        pickle.dump(save_object, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_parameter(file_dir):
    """
    Load the parameter was saved as pickled representation.

    Args:
        file_dir (str): file path

    Returns:
        Any: the parameter can be any type
    """
    with open(file_dir, 'rb') as f:
        output = pickle.load(f)
    return output

dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
    }
def replace_all(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text

def clean_text(text):
    text = replace_all(text, dict_map)
    pattern = r"[^a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+"
    text = re.sub(pattern, " ", text)
    text = " ".join(text.split())
    text = text.lower()
    return text

def flatten2DList(t):
    return [item for sublist in t for item in sublist]

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    device = get_device()
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1).to(device)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1).to(device)
    out = torch.mm(a_norm, b_norm.transpose(0, 1)).cpu()
    return out

def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)