import torch
import torch.nn as nn

from .task_vector import TaskVector


class TemporalModelFuser:
    """
    Temporal Model Fuser for merging checkpoints across time stages.
    Provides various strategies for extrapolating or fusing model parameters.
    """
    def __init__(self, base_model):
        """
        Initialize the Fuser.
        :param base_model: Pretrained base LLM used for calculating task vectors.
        """
        self.base_model = base_model

    def _get_task_vector(self, model):
        """Helper to calculate the task vector for a single model."""
        return TaskVector(pretrained_model=self.base_model, finetuned_model=model)

    def _get_diff_vector(self, model1, model2):
        """Helper to calculate the difference vector between two models (M2 - M1)."""
        return TaskVector(pretrained_model=model1, finetuned_model=model2)

    def extrapolate_linear_velocity(self, checkpoints, scaling_coefficient=1.0):
        """
        Linear extrapolation using the last two checkpoints (first-order velocity).
        M_pred = M_t + alpha * (M_t - M_{t-1})
        """
        if len(checkpoints) < 2:
            raise ValueError("Linear extrapolation requires at least 2 checkpoints.")
        
        phase_t_minus_1_model = checkpoints[-2]
        phase_t_model = checkpoints[-1]
        
        diff_vector = self._get_diff_vector(phase_t_minus_1_model, phase_t_model)
        merged_model = diff_vector.combine_with_pretrained_model(
            pretrained_model=phase_t_model, 
            scaling_coefficient=scaling_coefficient
        )
        return merged_model

    def extrapolate_moving_average_velocity(self, checkpoints, scaling_coefficient=1.0):
        """
        Moving average velocity based extrapolation for smoother predictions.
        V_avg = mean(M_i - M_{i-1} for i=1..t)
        M_pred = M_t + alpha * V_avg
        """
        if len(checkpoints) < 2:
            raise ValueError("Moving average velocity requires at least 2 checkpoints.")
        
        diff_vectors = []
        for i in range(len(checkpoints) - 1):
            vec = self._get_diff_vector(checkpoints[i], checkpoints[i+1])
            diff_vectors.append(vec)
            
        # Calculate average transition vector
        avg_diff_vector = diff_vectors[0]
        for i in range(1, len(diff_vectors)):
            avg_diff_vector += diff_vectors[i]
        avg_diff_vector *= (1.0 / len(diff_vectors))
        
        # Apply to the latest model
        latest_model = checkpoints[-1]
        merged_model = avg_diff_vector.combine_with_pretrained_model(
            pretrained_model=latest_model,
            scaling_coefficient=scaling_coefficient
        )
        return merged_model

    def extrapolate_with_acceleration(self, checkpoints, velocity_scale=1.0, acceleration_scale=0.5):
        """
        Extrapolation with second-order differences (acceleration) to capture non-linear trends.
        V_t = M_t - M_{t-1}
        A_t = V_t - V_{t-1} = (M_t - M_{t-1}) - (M_{t-1} - m_{t-2})
        M_pred = M_t + alpha * V_t + beta * A_t
        """
        if len(checkpoints) < 3:
            raise ValueError("Acceleration extrapolation requires at least 3 checkpoints.")
            
        m_t_minus_2, m_t_minus_1, m_t = checkpoints[-3:]
        
        # Latest velocity V_t
        velocity_vector = self._get_diff_vector(m_t_minus_1, m_t)
        
        # Acceleration A_t
        prev_velocity_vector = self._get_diff_vector(m_t_minus_2, m_t_minus_1)
        acceleration_vector = velocity_vector + (prev_velocity_vector * -1.0)

        # Combination
        # M_pred = M_t + alpha * V_t
        intermediate_model = velocity_vector.combine_with_pretrained_model(
            pretrained_model=m_t,
            scaling_coefficient=velocity_scale
        )
        # M_pred = (M_t + alpha*V_t) + beta * A_t
        final_model = acceleration_vector.combine_with_pretrained_model(
            pretrained_model=intermediate_model,
            scaling_coefficient=acceleration_scale
        )
        return final_model
    
    def fuse_exponential_decay(self, checkpoints, decay_rate=0.8):
        """
        Weighted fusion of all historical task vectors with exponential decay.
        Conservative but robust strategy.
        V_merged = sum(w_i * V_i) / sum(w_i) where w_i = decay_rate^(t-i)
        M_merged = M_base + V_merged
        """
        if not checkpoints:
            raise ValueError("Checkpoints list cannot be empty.")
        
        num_checkpoints = len(checkpoints)
        task_vectors = [self._get_task_vector(c) for c in checkpoints]
        
        # Calculate weights
        weights = [decay_rate ** (num_checkpoints - 1 - i) for i in range(num_checkpoints)]
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted sum
        weighted_sum_vector = task_vectors[0] * normalized_weights[0]
        for i in range(1, num_checkpoints):
            weighted_sum_vector += (task_vectors[i] * normalized_weights[i])
            
        # Apply to base model
        merged_model = weighted_sum_vector.combine_with_pretrained_model(self.base_model)
        return merged_model
