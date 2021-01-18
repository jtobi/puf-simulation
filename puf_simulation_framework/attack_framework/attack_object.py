from abc import ABC, abstractmethod



class AttackObject(ABC):
    @abstractmethod
    def get_summary_dict(self):
        """
        Has to return a dictionary with all relevant info about the attack
        :return:
        """
        pass



