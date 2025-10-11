"""add_last_accessed_timestamps

Revision ID: 2f3a4b5c6d7e
Revises: 1daae0df1866
Create Date: 2025-10-11 12:46:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '2f3a4b5c6d7e'
down_revision: Union[str, None] = '1daae0df1866'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add last_accessed timestamp columns to relevant tables for data cleanup feature.
    This supports issue #1335: cleanup_unused_data functionality.
    
    Tables affected:
    - document_chunks: stores text chunks from documents
    - entities: stores extracted entities from knowledge graph
    - summaries: stores generated summaries
    - associations: stores relationships between entities
    - metadata: stores metadata information
    """
    # Add last_accessed column to document_chunks table
    op.add_column('document_chunks', 
                  sa.Column('last_accessed', sa.DateTime(timezone=True), 
                           nullable=True, 
                           server_default=sa.text('CURRENT_TIMESTAMP')))
    
    # Add last_accessed column to entities table
    op.add_column('entities', 
                  sa.Column('last_accessed', sa.DateTime(timezone=True), 
                           nullable=True, 
                           server_default=sa.text('CURRENT_TIMESTAMP')))
    
    # Add last_accessed column to summaries table
    op.add_column('summaries', 
                  sa.Column('last_accessed', sa.DateTime(timezone=True), 
                           nullable=True, 
                           server_default=sa.text('CURRENT_TIMESTAMP')))
    
    # Add last_accessed column to associations table
    op.add_column('associations', 
                  sa.Column('last_accessed', sa.DateTime(timezone=True), 
                           nullable=True, 
                           server_default=sa.text('CURRENT_TIMESTAMP')))
    
    # Add last_accessed column to metadata table
    op.add_column('metadata', 
                  sa.Column('last_accessed', sa.DateTime(timezone=True), 
                           nullable=True, 
                           server_default=sa.text('CURRENT_TIMESTAMP')))
    
    # Create indexes on last_accessed columns for efficient cleanup queries
    op.create_index('ix_document_chunks_last_accessed', 'document_chunks', ['last_accessed'])
    op.create_index('ix_entities_last_accessed', 'entities', ['last_accessed'])
    op.create_index('ix_summaries_last_accessed', 'summaries', ['last_accessed'])
    op.create_index('ix_associations_last_accessed', 'associations', ['last_accessed'])
    op.create_index('ix_metadata_last_accessed', 'metadata', ['last_accessed'])


def downgrade() -> None:
    """
    Remove last_accessed timestamp columns and indexes from all tables.
    """
    # Drop indexes first
    op.drop_index('ix_metadata_last_accessed', table_name='metadata')
    op.drop_index('ix_associations_last_accessed', table_name='associations')
    op.drop_index('ix_summaries_last_accessed', table_name='summaries')
    op.drop_index('ix_entities_last_accessed', table_name='entities')
    op.drop_index('ix_document_chunks_last_accessed', table_name='document_chunks')
    
    # Then drop columns
    op.drop_column('metadata', 'last_accessed')
    op.drop_column('associations', 'last_accessed')
    op.drop_column('summaries', 'last_accessed')
    op.drop_column('entities', 'last_accessed')
    op.drop_column('document_chunks', 'last_accessed')
