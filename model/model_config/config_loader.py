import yaml
class DotDict(dict):
    """Dictionary subclass that allows for dot notation access to attributes"""
    def __getattr__(self, attr):
        return self.get(attr)
    
    def __setattr__(self, attr, value):
        self[attr] = value
    
    def __delattr__(self, attr):
        del self[attr]


def yaml_to_dotdict(file_path):
    """Load a YAML file and convert it to a DotDict"""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return _convert_dict_to_dotdict(data)


def _convert_dict_to_dotdict(data):
    """Recursively convert dictionaries to DotDict"""
    if isinstance(data, dict):
        return DotDict({k: _convert_dict_to_dotdict(v) for k, v in data.items()})
    elif isinstance(data, list):
        return [_convert_dict_to_dotdict(item) for item in data]
    else:
        return data