import re
import json

class FileLoader(object):
    def __init__(self, file_path):
        super(FileLoader, self).__init__()
        self.path = file_path

    def load(self):
        with open(self.path, 'r') as f:
            raw_data = f.readlines()
        return raw_data

class JsonFileLoader(object):
    def __init__(self, file_path):
        super(JsonFileLoader, self).__init__()
        self.path = file_path

    def load(self):
        with open(self.path, 'r') as f:
            raw_dataset = json.load(f)
        return raw_dataset

# todo(zqzhang): updated in TPv7
class PTFileLoader(object):
    def __init__(self, file_path):
        super(PTFileLoader, self).__init__()
        self.path = file_path

    def load(self):
        import torch as t
        import numpy as np
        Content = t.load(self.path, map_location = 'cpu')
        if Content.__class__ == t.Tensor:
            Content = Content.cpu()
        Content = np.array(Content)
        return Content

class NpyFileLoader(object):
    def __init__(self, file_path):
        super(NpyFileLoader, self).__init__()
        self.path = file_path

    def load(self):
        import numpy as np
        Content = np.load(self.path)
        return Content

class NpzFileLoader(object):
    def __init__(self, file_path):
        super(NpzFileLoader, self).__init__()
        self.path = file_path

    def load(self):
        import numpy as np
        Content = np.load(self.path)
        return Content

##############################################
# Parse files for different dataset files
##############################################
class BasicFileParser(object):
    def __init__(self):
        super(BasicFileParser, self).__init__()

    def _parse_line(self, line):
        raise NotImplementedError(
            "Line parser not implemented."
        )

    def parse_file(self, raw_data):
        Dataset = []
        for line in raw_data:
            data = self._parse_line(line)
            Dataset.append(data)
        return Dataset

class HIVFileParser(BasicFileParser):
    def __init__(self):
        super(HIVFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}


class BBBPFileParser(BasicFileParser):
    def __init__(self):
        super(BBBPFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}


class BACEFileParser(BasicFileParser):
    def __init__(self):
        super(BACEFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}


class QM9FileParser(BasicFileParser):
    def __init__(self):
        super(QM9FileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        return {'SMILES': SMILES, 'Value': Value}


class FreeSolvFileParser(BasicFileParser):
    def __init__(self):
        super(FreeSolvFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}


class LipopFileParser(BasicFileParser):
    def __init__(self):
        super(LipopFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}


class MalariaFileParser(BasicFileParser):
    def __init__(self):
        super(MalariaFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}


class CEPFileParser(BasicFileParser):
    def __init__(self):
        super(CEPFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}


class SHP2FileParser(BasicFileParser):
    def __init__(self):
        super(SHP2FileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}


class Tox21FileParser(BasicFileParser):
    def __init__(self):
        super(Tox21FileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        for i in range(len(Value)):
            value = Value[i]
            if value == '':
                Value[i] = '-1'
        return {'SMILES': SMILES, 'Value': Value}


class ToxcastFileParser(BasicFileParser):
    def __init__(self):
        super(ToxcastFileParser, self).__init__()

    def _parse_line(self, line):
        # Convert '1.0/0.0' to '1/0'
        # Convert missing value '' to '-1'
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        for i in range(len(Value)):
            value = Value[i]
            if value == '':
                Value[i] = '-1'
            elif value == '0.0':
                Value[i] = '0'
            elif value == '1.0':
                Value[i] = '1'
        return {'SMILES': SMILES, 'Value': Value}


class MUVFileParser(BasicFileParser):
    def __init__(self):
        super(MUVFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        for i in range(len(Value)):
            value = Value[i]
            if value == '':
                Value[i] = '-1'
        return {"SMILES": SMILES, 'Value': Value}


class ClinToxFileParser(BasicFileParser):
    def __init__(self):
        super(ClinToxFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        return {'SMILES': SMILES, 'Value': Value}


class SIDERFileParser(BasicFileParser):
    def __init__(self):
        super(SIDERFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1:]
        Value[-1] = re.split('\n', Value[-1])[0]
        return {'SMILES': SMILES, 'Value': Value}


class ESOLFileParser(BasicFileParser):
    def __init__(self):
        super(ESOLFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',', line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}
################################################