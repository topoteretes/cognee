/**
 * Type definitions for Cognee API responses
 * Based on Cognee SDK data point model and graph API
 */

/**
 * Cognee DataPoint representation from API
 * Corresponds to: cognee/infrastructure/engine/models/DataPoint.py
 */
export interface CogneeDataPoint {
  id: string;                              // UUID
  label: string;                           // Display name
  type: string;                            // Node type (Entity, EntityType, DocumentChunk, etc.)
  properties?: Record<string, any>;        // Additional metadata
}

/**
 * Cognee Edge representation from API
 * Corresponds to: cognee/infrastructure/engine/models/Edge.py
 */
export interface CogneeEdge {
  source: string;                          // Source node UUID
  target: string;                          // Target node UUID
  label: string;                           // Relationship type
  weight?: number;                         // Optional weight
  weights?: Record<string, number>;        // Optional multiple weights
  properties?: Record<string, any>;        // Additional properties
}

/**
 * Cognee Graph API response format
 * From: /api/v1/datasets/{dataset_id}/graph
 */
export interface CogneeGraphResponse {
  nodes: CogneeDataPoint[];
  edges: CogneeEdge[];
}

/**
 * Cognee API error response
 */
export interface CogneeAPIError {
  detail: string;
  status_code: number;
}
