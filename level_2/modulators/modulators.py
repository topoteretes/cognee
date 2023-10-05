import numpy as np


class DifferentiableLayer:
    def __init__(self, attention_modulators: dict):
        self.weights = {modulator: 1.0 for modulator in attention_modulators}
        self.learning_rate = 0.1
        self.regularization_lambda = 0.01
        self.weight_decay = 0.99

    async def adjust_weights(self, feedbacks: list[float]):
        """
        Adjusts the weights of the attention modulators based on user feedbacks.

        Parameters:
        - feedbacks: A list of feedback scores (between 0 and 1).
        """
        avg_feedback = np.mean(feedbacks)
        feedback_diff = 1.0 - avg_feedback

        # Adjust weights based on average feedback
        for modulator in self.weights:
            self.weights[modulator] += self.learning_rate * (-feedback_diff) - self.regularization_lambda * \
                                       self.weights[modulator]
            self.weights[modulator] *= self.weight_decay

        # Decaying the learning rate
        self.learning_rate *= 0.99

    async def get_weights(self):
        return self.weights

