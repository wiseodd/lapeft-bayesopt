from lapeft_bayesopt.problems.prompting import PromptBuilder


class MyPromptBuilder(PromptBuilder):
    def __init__(self, kind: str):
        self.kind = kind

    def get_prompt(self, x: str, obj_str: str) -> str:
        if self.kind == "completion":
            return f"The estimated {obj_str} of the molecule {x} is: "
        elif self.kind == "just-smiles":
            return x
        else:
            return NotImplementedError
