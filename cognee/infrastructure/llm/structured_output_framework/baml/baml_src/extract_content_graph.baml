class Node {
    id string
    name string
    type string
    description string
    @@dynamic
}

/// doc string for edge
class Edge {
    /// doc string for source_node_id
    source_node_id string
    target_node_id string
    relationship_name string
}

class KnowledgeGraph {
    nodes (Node @stream.done)[]
    edges Edge[]
}

// Summarization classes
class SummarizedContent {
    summary string
    description string
}

class SummarizedFunction {
    name string
    description string
    inputs string[]?
    outputs string[]?
    decorators string[]?
}

class SummarizedClass {
    name string
    description string
    methods SummarizedFunction[]?
    decorators string[]?
}

class SummarizedCode {
    high_level_summary string
    key_features string[]
    imports string[]
    constants string[]
    classes SummarizedClass[]
    functions SummarizedFunction[]
    workflow_description string?
}

class DynamicKnowledgeGraph {
    @@dynamic
}


// Simple template for basic extraction (fast, good quality)
template_string ExtractContentGraphPrompt() #"
    You are an advanced algorithm that extracts structured data into a knowledge graph.

    - **Nodes**: Entities/concepts (like Wikipedia articles).
    - **Edges**: Relationships (like Wikipedia links). Use snake_case (e.g., `acted_in`).

    **Rules:**

    1. **Node Labeling & IDs**
    - Use basic types only (e.g., "Person", "Date", "Organization").
    - Avoid overly specific or generic terms (e.g., no "Mathematician" or "Entity").
    - Node IDs must be human-readable names from the text (no numbers).

    2. **Dates & Numbers**
    - Label dates as **"Date"** in "YYYY-MM-DD" format (use available parts if incomplete).
    - Properties are key-value pairs; do not use escaped quotes.

    3. **Coreference Resolution**
    - Use a single, complete identifier for each entity (e.g., always "John Doe" not "Joe" or "he").

    4. **Relationship Labels**:
    - Use descriptive, lowercase, snake_case names for edges.
    - *Example*: born_in, married_to, invented_by.
    - Avoid vague or generic labels like isA, relatesTo, has.
    - Avoid duplicated relationships like produces, produced by.

    5. **Strict Compliance**
    - Follow these rules exactly. Non-compliance results in termination.
"#

// Summarization prompt template
template_string SummarizeContentPrompt() #"
    You are a top-tier summarization engine. Your task is to summarize text and make it versatile.
    Be brief and concise, but keep the important information and the subject.
    Use synonym words where possible in order to change the wording but keep the meaning.
"#

// Code summarization prompt template
template_string SummarizeCodePrompt() #"
    You are an expert code analyst. Analyze the provided source code and extract key information:

    1. Provide a high-level summary of what the code does
    2. List key features and functionality
    3. Identify imports and dependencies
    4. List constants and global variables
    5. Summarize classes with their methods
    6. Summarize standalone functions
    7. Describe the overall workflow if applicable

    Be precise and technical while remaining clear and concise.
"#

// Detailed template for complex extraction (slower, higher quality)
template_string DetailedExtractContentGraphPrompt() #"
    You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
    **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
    **Edges** represent relationships between concepts. They're akin to Wikipedia links.

    The aim is to achieve simplicity and clarity in the knowledge graph.

    # 1. Labeling Nodes
    **Consistency**: Ensure you use basic or elementary types for node labels.
    - For example, when you identify an entity representing a person, always label it as **"Person"**.
    - Avoid using more specific terms like "Mathematician" or "Scientist", keep those as "profession" property.
    - Don't use too generic terms like "Entity".
    **Node IDs**: Never utilize integers as node IDs.
    - Node IDs should be names or human-readable identifiers found in the text.

    # 2. Handling Numerical Data and Dates
    - For example, when you identify an entity representing a date, make sure it has type **"Date"**.
    - Extract the date in the format "YYYY-MM-DD"
    - If not possible to extract the whole date, extract month or year, or both if available.
    - **Property Format**: Properties must be in a key-value format.
    - **Quotation Marks**: Never use escaped single or double quotes within property values.
    - **Naming Convention**: Use snake_case for relationship names, e.g., `acted_in`.

    # 3. Coreference Resolution
    - **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
    If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
    always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the Person's ID.
    Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.

    # 4. Strict Compliance
    Adhere to the rules strictly. Non-compliance will result in termination.
"#

// Guided template with step-by-step instructions
template_string GuidedExtractContentGraphPrompt() #"
    You are an advanced algorithm designed to extract structured information to build a clean, consistent, and human-readable knowledge graph.

    **Objective**:
    - Nodes represent entities and concepts, similar to Wikipedia articles.
    - Edges represent typed relationships between nodes, similar to Wikipedia hyperlinks.
    - The graph must be clear, minimal, consistent, and semantically precise.

    **Node Guidelines**:

    1. **Label Consistency**:
       - Use consistent, basic types for all node labels.
       - Do not switch between granular or vague labels for the same kind of entity.
       - Pick one label for each category and apply it uniformly.
       - Each entity type should be in a singular form and in a case of multiple words separated by whitespaces

    2. **Node Identifiers**:
       - Node IDs must be human-readable and derived directly from the text.
       - Prefer full names and canonical terms.
       - Never use integers or autogenerated IDs.
       - *Example*: Use "Marie Curie", "Theory of Evolution", "Google".

    3. **Coreference Resolution**:
       - Maintain one consistent node ID for each real-world entity.
       - Resolve aliases, acronyms, and pronouns to the most complete form.
       - *Example*: Always use "John Doe" even if later referred to as "Doe" or "he".

    **Edge Guidelines**:

    4. **Relationship Labels**:
       - Use descriptive, lowercase, snake_case names for edges.
       - *Example*: born_in, married_to, invented_by.
       - Avoid vague or generic labels like isA, relatesTo, has.

    5. **Relationship Direction**:
       - Edges must be directional and logically consistent.
       - *Example*:
         - "Marie Curie" —[born_in]→ "Warsaw"
         - "Radioactivity" —[discovered_by]→ "Marie Curie"

    **Compliance**:
    Strict adherence to these guidelines is required. Any deviation will result in immediate termination of the task.
"#

// Strict template with zero-tolerance rules
template_string StrictExtractContentGraphPrompt() #"
    You are a top-tier algorithm for **extracting structured information** from unstructured text to build a **knowledge graph**.

    Your primary goal is to extract:
    - **Nodes**: Representing **entities** and **concepts** (like Wikipedia nodes).
    - **Edges**: Representing **relationships** between those concepts (like Wikipedia links).

    The resulting knowledge graph must be **simple, consistent, and human-readable**.

    ## 1. Node Labeling and Identification

    ### Node Types
    Use **basic atomic types** for node labels. Always prefer general types over specific roles or professions:
    - "Person" for any human.
    - "Organization" for companies, institutions, etc.
    - "Location" for geographic or place entities.
    - "Date" for any temporal expression.
    - "Event" for historical or scheduled occurrences.
    - "Work" for books, films, artworks, or research papers.
    - "Concept" for abstract notions or ideas.

    ### Node IDs
    - Always assign **human-readable and unambiguous identifiers**.
    - Never use numeric or autogenerated IDs.
    - Prioritize **most complete form** of entity names for consistency.

    ## 2. Relationship Handling
    - Use **snake_case** for all relationship (edge) types.
    - Keep relationship types semantically clear and consistent.
    - Avoid vague relation names like "related_to" unless no better alternative exists.

    ## 3. Strict Compliance
    Follow all rules exactly. Any deviation may lead to rejection or incorrect graph construction.
"#

// OpenAI client with environment model selection
client<llm> OpenAI {
    provider openai
    options {
        model client_registry.model
        api_key client_registry.api_key
    }
}



// Function that returns raw structured output (for custom objects - to be handled in Python)
function ExtractContentGraphGeneric(
    content: string,
    mode: "simple" | "base" | "guided" | "strict" | "custom"?,
    custom_prompt_content: string?
) -> KnowledgeGraph {
    client OpenAI

    prompt #"
        {% if mode == "base" %}
            {{ DetailedExtractContentGraphPrompt() }}
        {% elif mode == "guided" %}
            {{ GuidedExtractContentGraphPrompt() }}
        {% elif mode == "strict" %}
            {{ StrictExtractContentGraphPrompt() }}
        {% elif mode == "custom" and custom_prompt_content %}
            {{ custom_prompt_content }}
        {% else %}
            {{ ExtractContentGraphPrompt() }}
        {% endif %}

        {{ ctx.output_format(prefix="Answer in this schema:\n") }}

        Before answering, briefly describe what you'll extract from the text, then provide the structured output.

        Example format:
        I'll extract the main entities and their relationships from this text...

        { ... }

        {{ _.role('user') }}
        {{ content }}
    "#
}

// Backward-compatible function specifically for KnowledgeGraph
function ExtractDynamicContentGraph(
    content: string,
    mode: "simple" | "base" | "guided" | "strict" | "custom"?,
    custom_prompt_content: string?
) -> DynamicKnowledgeGraph {
    client OpenAI

    prompt #"
        {% if mode == "base" %}
            {{ DetailedExtractContentGraphPrompt() }}
        {% elif mode == "guided" %}
            {{ GuidedExtractContentGraphPrompt() }}
        {% elif mode == "strict" %}
            {{ StrictExtractContentGraphPrompt() }}
        {% elif mode == "custom" and custom_prompt_content %}
            {{ custom_prompt_content }}
        {% else %}
            {{ ExtractContentGraphPrompt() }}
        {% endif %}

        {{ ctx.output_format(prefix="Answer in this schema:\n") }}

        Before answering, briefly describe what you'll extract from the text, then provide the structured output.

        Example format:
        I'll extract the main entities and their relationships from this text...

        { ... }

        {{ _.role('user') }}
        {{ content }}
    "#
}


// Summarization functions
function SummarizeContent(content: string) -> SummarizedContent {
    client OpenAI

    prompt #"
        {{ SummarizeContentPrompt() }}

        {{ ctx.output_format(prefix="Answer in this schema:\n") }}

        {{ _.role('user') }}
        {{ content }}
    "#
}

function SummarizeCode(content: string) -> SummarizedCode {
    client OpenAI

    prompt #"
        {{ SummarizeCodePrompt() }}

        {{ ctx.output_format(prefix="Answer in this schema:\n") }}

        {{ _.role('user') }}
        {{ content }}
    "#
}

test ExtractStrictExample {
    functions [ExtractContentGraphGeneric]
    args {
        content #"
            The Python programming language was created by Guido van Rossum in 1991.
        "#
        mode "strict"
    }
}
