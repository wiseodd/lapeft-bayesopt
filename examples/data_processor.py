from lapeft_bayesopt.problems.data_processor import DataProcessor
from typing import *


class RedoxDataProcessor(DataProcessor):
    """
    RangeIndex: 1407 entries, 0 to 1406
    Data columns (total 6 columns):
    #   Column                 Non-Null Count  Dtype
    --  ------                 --------------  -----
    0   Entry Number           1407 non-null   int64
    1   File Name              1407 non-null   object
    2   SMILES                 1407 non-null   object
    3   Ered                   1407 non-null   float64
    4   HOMO                   1407 non-null   float64
    5   Gsol                   1407 non-null   float64
    6   Absorption Wavelength  1407 non-null   float64
    dtypes: float64(4), int64(1), object(2)
    memory usage: 77.1+ KB

    Objective: Minimize Ered (secondary objective: minimize Gsol)
    """
    def __init__(self, prompt_builder, tokenizer):
        super().__init__(prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer)
        self.x_col = 'SMILES'
        self.target_col = 'Ered'
        self.obj_str = 'redox potential'
        self.maximization = False

    def _get_columns_to_remove(self) -> List[str]:
        return ['Entry Number', 'File Name', 'SMILES', 'HOMO', 'Ered', 'Gsol', 'Absorption Wavelength']


class MultiRedoxDataProcessor(DataProcessor):
    """
    RangeIndex: 1407 entries, 0 to 1406
    Data columns (total 6 columns):
    #   Column                 Non-Null Count  Dtype
    --  ------                 --------------  -----
    0   Entry Number           1407 non-null   int64
    1   File Name              1407 non-null   object
    2   SMILES                 1407 non-null   object
    3   Ered                   1407 non-null   float64
    4   HOMO                   1407 non-null   float64
    5   Gsol                   1407 non-null   float64
    6   Absorption Wavelength  1407 non-null   float64
    dtypes: float64(4), int64(1), object(2)
    memory usage: 77.1+ KB

    Objective: Minimize Ered, minimize Gsol
    """
    def __init__(self, prompt_builder, tokenizer):
        super().__init__(prompt_builder=prompt_builder, num_outputs=2, tokenizer=tokenizer)
        self.x_col = 'SMILES'
        self.target_col = ['Ered', 'Gsol']
        self.obj_str = 'redox potential and solvation energy'
        self.maximization = [False, False]

    def _get_columns_to_remove(self) -> List[str]:
        return ['Entry Number', 'File Name', 'SMILES', 'HOMO', 'Ered', 'Gsol', 'Absorption Wavelength']
