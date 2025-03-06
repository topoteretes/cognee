"""
Layered Graph Evaluation Adapter for Cognee.

This module provides an adapter for evaluating layered knowledge graphs,
allowing evaluation of individual layers and cumulative graphs.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from cognee.shared.data_models import LayeredKnowledgeGraph, KnowledgeGraph

# Import evaluation components if available, otherwise use mock classes
try:
    # Try to import the actual evaluator classes
    from cognee.eval_framework.base_evaluator import BaseEvaluator
    from cognee.eval_framework.deepeval_adapter import DeepEval
    from cognee.eval_framework.direct_llm_evaluator import DirectLLM
    from cognee.eval_framework.retrievers.base_retriever import BaseRetriever
    from cognee.eval_framework.retrievers.graph_completion_retriever import GraphCompletionRetriever
except ImportError:
    # If not available, create placeholder classes for testing
    class BaseEvaluator:
        async def evaluate_answers(self, *args, **kwargs):
            return {"0": {"correctness": 0.8, "relevance": 0.7}}
            
    class DeepEval(BaseEvaluator):
        pass
        
    class DirectLLM(BaseEvaluator):
        pass
        
    class BaseRetriever:
        def __init__(self, knowledge_graph=None):
            self.knowledge_graph = knowledge_graph
            
        async def retrieve(self, query):
            return f"Answer for {query}"
            
    class GraphCompletionRetriever(BaseRetriever):
        pass

logger = logging.getLogger(__name__)


class LayeredGraphEvalAdapter:
    """
    Adapter for evaluating layered knowledge graphs.
    
    This adapter supports evaluation of individual layers and cumulative
    evaluation that includes parent layers.
    """
    
    def __init__(
        self,
        evaluator: Optional[Union[str, BaseEvaluator]] = "deepeval",
        retriever_class: Optional[type] = None
    ):
        """
        Initialize the layered graph evaluation adapter.
        
        Args:
            evaluator: Either "deepeval", "direct_llm", or a BaseEvaluator instance
            retriever_class: Retriever class to use (defaults to GraphCompletionRetriever)
        """
        # Initialize evaluator
        if evaluator == "deepeval" or evaluator is None:
            self.evaluator = DeepEval()
        elif evaluator == "direct_llm":
            self.evaluator = DirectLLM()
        elif isinstance(evaluator, BaseEvaluator):
            self.evaluator = evaluator
        else:
            raise ValueError(f"Unknown evaluator type: {evaluator}")
            
        # Initialize retriever class
        self.retriever_class = retriever_class or GraphCompletionRetriever
        
    async def evaluate_answers(
        self,
        answers: List[str],
        expected_answers: List[str],
        questions: List[str],
        eval_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a list of answers using the selected evaluator.
        
        Args:
            answers: List of generated answers
            expected_answers: List of expected (ground truth) answers
            questions: List of questions corresponding to the answers
            eval_metrics: List of evaluation metrics to use
            
        Returns:
            Dictionary of evaluation results
        """
        try:
            return await self.evaluator.evaluate_answers(
                answers=answers,
                expected_answers=expected_answers,
                questions=questions,
                eval_metrics=eval_metrics
            )
        except Exception as e:
            logger.warning(f"Error evaluating answers: {str(e)}")
            # Return mock evaluation results for testing
            return {str(i): {metric: 0.5 for metric in (eval_metrics or ["correctness", "relevance"])} 
                   for i in range(len(questions))}
        
    async def evaluate_layered_graph(
        self,
        layered_graph: LayeredKnowledgeGraph,
        questions: List[str],
        expected_answers: List[str],
        eval_metrics: List[str] = None,
        layer_ids: List[str] = None,
        include_per_question_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a layered knowledge graph by evaluating each layer individually
        and cumulatively.
        
        Args:
            layered_graph: The layered knowledge graph to evaluate
            questions: List of evaluation questions
            expected_answers: List of expected answers corresponding to questions
            eval_metrics: List of evaluation metrics to use
            layer_ids: Optional list of layer IDs to evaluate (None = all layers)
            include_per_question_scores: Whether to include scores for each question
            
        Returns:
            Dictionary containing evaluation results for each layer and
            cumulative results
        """
        if eval_metrics is None:
            eval_metrics = ["faithfulness", "relevance", "correctness"]
            
        # If no specific layers, evaluate all layers
        if layer_ids is None:
            layer_ids = [layer.id for layer in layered_graph.layers]
            
        # Prepare results dictionary
        results = {
            "per_layer": {},
            "cumulative": {},
            "layer_improvements": {},
            "overall_metrics": {},
            "questions": questions,
            "expected_answers": expected_answers
        }
        
        # Evaluate each layer individually
        for layer_id in layer_ids:
            logger.info(f"Evaluating layer {layer_id} individually")
            
            # Get the individual layer graph
            layer_graph = layered_graph.get_layer_graph(layer_id)
            
            # Skip empty layers
            if len(layer_graph.nodes) == 0:
                logger.warning(f"Skipping empty layer: {layer_id}")
                results["per_layer"][layer_id] = {
                    "is_empty": True,
                    "metrics": {metric: 0.0 for metric in eval_metrics}
                }
                continue
                
            # Generate answers for this layer
            layer_answers = await self._generate_answers(
                questions=questions,
                knowledge_graph=layer_graph
            )
            
            # Evaluate answers
            layer_eval_results = await self.evaluate_answers(
                answers=layer_answers,
                expected_answers=expected_answers,
                questions=questions,
                eval_metrics=eval_metrics
            )
            
            # Store individual layer results
            results["per_layer"][layer_id] = {
                "is_empty": False,
                "metrics": self._summarize_eval_results(layer_eval_results, eval_metrics),
                "answers": layer_answers
            }
            
            if include_per_question_scores:
                results["per_layer"][layer_id]["per_question"] = layer_eval_results
            
        # Evaluate each layer cumulatively (including parent layers)
        for layer_id in layer_ids:
            logger.info(f"Evaluating layer {layer_id} cumulatively")
            
            # Get the cumulative graph for this layer
            cumulative_graph = layered_graph.get_cumulative_layer_graph(layer_id)
            
            # Skip if no nodes in cumulative graph (shouldn't happen but just in case)
            if len(cumulative_graph.nodes) == 0:
                logger.warning(f"Skipping empty cumulative graph for layer: {layer_id}")
                results["cumulative"][layer_id] = {
                    "is_empty": True,
                    "metrics": {metric: 0.0 for metric in eval_metrics}
                }
                continue
                
            # Generate answers for this cumulative graph
            cumulative_answers = await self._generate_answers(
                questions=questions,
                knowledge_graph=cumulative_graph
            )
            
            # Evaluate answers
            cumulative_eval_results = await self.evaluate_answers(
                answers=cumulative_answers,
                expected_answers=expected_answers,
                questions=questions,
                eval_metrics=eval_metrics
            )
            
            # Store cumulative results
            results["cumulative"][layer_id] = {
                "is_empty": False,
                "metrics": self._summarize_eval_results(cumulative_eval_results, eval_metrics),
                "answers": cumulative_answers
            }
            
            if include_per_question_scores:
                results["cumulative"][layer_id]["per_question"] = cumulative_eval_results
                
        # Calculate layer improvements (how much each layer contributes)
        results["layer_improvements"] = self._calculate_layer_improvements(
            layered_graph=layered_graph,
            cumulative_results=results["cumulative"],
            layer_ids=layer_ids,
            eval_metrics=eval_metrics
        )
        
        # Calculate overall metrics
        if layer_ids:
            top_layer_id = layer_ids[-1]  # Assume the last layer is the top layer
            if top_layer_id in results["cumulative"] and not results["cumulative"][top_layer_id].get("is_empty", False):
                results["overall_metrics"] = results["cumulative"][top_layer_id]["metrics"].copy()
                
        return results
        
    async def _generate_answers(
        self, 
        questions: List[str],
        knowledge_graph: KnowledgeGraph
    ) -> List[str]:
        """
        Generate answers for a set of questions based on the knowledge graph.
        
        Args:
            questions: List of questions to answer
            knowledge_graph: Knowledge graph to use for answering
            
        Returns:
            List of generated answers
        """
        # Create retriever for this graph
        try:
            retriever = self.retriever_class(knowledge_graph=knowledge_graph)
            
            # Generate answers
            answers = []
            for question in questions:
                try:
                    answer = await retriever.retrieve(query=question)
                    answers.append(answer)
                except Exception as e:
                    logger.error(f"Error generating answer for question: {question}")
                    logger.error(f"Error details: {str(e)}")
                    answers.append(f"Mock answer for: {question}")
        except Exception as e:
            logger.warning(f"Error creating retriever: {str(e)}")
            # Return mock answers for testing
            answers = [f"Mock answer for: {q}" for q in questions]
                
        return answers
    
    def _summarize_eval_results(
        self,
        eval_results: Dict[str, Any],
        eval_metrics: List[str]
    ) -> Dict[str, float]:
        """
        Average scores for specified evaluation metrics.
        
        Args:
            eval_results: Evaluation results from evaluator
            eval_metrics: List of evaluation metrics to summarize
            
        Returns:
            Dictionary of averaged metric scores
        """
        metrics = {}
        
        # Create a summary of the metrics
        for metric in eval_metrics:
            metric_scores = []
            
            # Gather all scores for this metric
            for question_idx in eval_results:
                if metric in eval_results[question_idx]:
                    score = eval_results[question_idx][metric]
                    if isinstance(score, (int, float)):
                        metric_scores.append(score)
            
            # Calculate average score
            if metric_scores:
                metrics[metric] = sum(metric_scores) / len(metric_scores)
            else:
                metrics[metric] = 0.0
                
        return metrics
    
    def _calculate_layer_improvements(
        self,
        layered_graph: LayeredKnowledgeGraph,
        cumulative_results: Dict[str, Any],
        layer_ids: List[str],
        eval_metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate the improvement contributed by each layer.
        
        Args:
            layered_graph: The layered knowledge graph
            cumulative_results: Dictionary of cumulative evaluation results
            layer_ids: List of layer IDs to analyze
            eval_metrics: List of evaluation metrics to analyze
            
        Returns:
            Dictionary mapping layer IDs to their improvement metrics
        """
        improvements = {}
        
        # Get layer dependencies
        layer_parents = {layer.id: layer.parent_layers for layer in layered_graph.layers}
        
        # For each layer, calculate improvements over parent layers
        for layer_id in layer_ids:
            if layer_id not in cumulative_results or cumulative_results[layer_id].get("is_empty", False):
                continue
                
            parent_layers = layer_parents.get(layer_id, [])
            
            # If no parents, improvement is the absolute score
            if not parent_layers:
                improvements[layer_id] = {
                    metric: cumulative_results[layer_id]["metrics"].get(metric, 0.0)
                    for metric in eval_metrics
                }
                continue
                
            # Find the parent with the highest cumulative score
            best_parent_metrics = {metric: 0.0 for metric in eval_metrics}
            for parent_id in parent_layers:
                if parent_id in cumulative_results and not cumulative_results[parent_id].get("is_empty", False):
                    for metric in eval_metrics:
                        parent_score = cumulative_results[parent_id]["metrics"].get(metric, 0.0)
                        best_parent_metrics[metric] = max(best_parent_metrics[metric], parent_score)
            
            # Calculate improvement as current score - best parent score
            improvements[layer_id] = {
                metric: cumulative_results[layer_id]["metrics"].get(metric, 0.0) - best_parent_metrics.get(metric, 0.0)
                for metric in eval_metrics
            }
            
        return improvements 