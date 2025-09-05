// Content classification data models - matching shared/data_models.py
class TextContent {
    type string
    subclass string[]
}

class AudioContent {
    type string
    subclass string[]
}

class ImageContent {
    type string
    subclass string[]
}

class VideoContent {
    type string
    subclass string[]
}

class MultimediaContent {
    type string
    subclass string[]
}

class Model3DContent {
    type string
    subclass string[]
}

class ProceduralContent {
    type string
    subclass string[]
}

class ContentLabel {
    content_type "text" | "audio" | "image" | "video" | "multimedia" | "3d_model" | "procedural"
    type string
    subclass string[]
}

class DefaultContentPrediction {
    label ContentLabel
}

// Content classification prompt template
template_string ClassifyContentPrompt() #"
    You are a classification engine and should classify content. Make sure to use one of the existing classification options and not invent your own.

    Classify the content into one of these main categories and their relevant subclasses:

    **TEXT CONTENT** (content_type: "text"):
    - type: "TEXTUAL_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    - subclass options: ["Articles, essays, and reports", "Books and manuscripts", "News stories and blog posts", "Research papers and academic publications", "Social media posts and comments", "Website content and product descriptions", "Personal narratives and stories", "Spreadsheets and tables", "Forms and surveys", "Databases and CSV files", "Source code in various programming languages", "Shell commands and scripts", "Markup languages (HTML, XML)", "Stylesheets (CSS) and configuration files (YAML, JSON, INI)", "Chat transcripts and messaging history", "Customer service logs and interactions", "Conversational AI training data", "Textbook content and lecture notes", "Exam questions and academic exercises", "E-learning course materials", "Poetry and prose", "Scripts for plays, movies, and television", "Song lyrics", "Manuals and user guides", "Technical specifications and API documentation", "Helpdesk articles and FAQs", "Contracts and agreements", "Laws, regulations, and legal case documents", "Policy documents and compliance materials", "Clinical trial reports", "Patient records and case notes", "Scientific journal articles", "Financial reports and statements", "Business plans and proposals", "Market research and analysis reports", "Ad copies and marketing slogans", "Product catalogs and brochures", "Press releases and promotional content", "Professional and formal correspondence", "Personal emails and letters", "Image and video captions", "Annotations and metadata for various media", "Vocabulary lists and grammar rules", "Language exercises and quizzes", "Other types of text data"]

    **AUDIO CONTENT** (content_type: "audio"):
    - type: "AUDIO_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    - subclass options: ["Music tracks and albums", "Podcasts and radio broadcasts", "Audiobooks and audio guides", "Recorded interviews and speeches", "Sound effects and ambient sounds", "Other types of audio recordings"]

    **IMAGE CONTENT** (content_type: "image"):
    - type: "IMAGE_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    - subclass options: ["Photographs and digital images", "Illustrations, diagrams, and charts", "Infographics and visual data representations", "Artwork and paintings", "Screenshots and graphical user interfaces", "Other types of images"]

    **VIDEO CONTENT** (content_type: "video"):
    - type: "VIDEO_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    - subclass options: ["Movies and short films", "Documentaries and educational videos", "Video tutorials and how-to guides", "Animated features and cartoons", "Live event recordings and sports broadcasts", "Other types of video content"]

    **MULTIMEDIA CONTENT** (content_type: "multimedia"):
    - type: "MULTIMEDIA_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    - subclass options: ["Interactive web content and games", "Virtual reality (VR) and augmented reality (AR) experiences", "Mixed media presentations and slide decks", "E-learning modules with integrated multimedia", "Digital exhibitions and virtual tours", "Other types of multimedia content"]

    **3D MODEL CONTENT** (content_type: "3d_model"):
    - type: "3D_MODEL_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    - subclass options: ["Architectural renderings and building plans", "Product design models and prototypes", "3D animations and character models", "Scientific simulations and visualizations", "Virtual objects for AR/VR applications", "Other types of 3D models"]

    **PROCEDURAL CONTENT** (content_type: "procedural"):
    - type: "PROCEDURAL_DOCUMENTS_USED_FOR_GENERAL_PURPOSES"
    - subclass options: ["Tutorials and step-by-step guides", "Workflow and process descriptions", "Simulation and training exercises", "Recipes and crafting instructions", "Other types of procedural content"]

    Select the most appropriate content_type, type, and relevant subclasses.
"#

// OpenAI client defined once for all BAML files

// Classification function
function ExtractCategories(content: string) -> DefaultContentPrediction {
    client OpenAI

    prompt #"
        {{ ClassifyContentPrompt() }}

        {{ ctx.output_format(prefix="Answer in this schema:\n") }}

        {{ _.role('user') }}
        {{ content }}
    "#
}

// Test case for classification
test ExtractCategoriesExample {
    functions [ExtractCategories]
    args {
        content #"
            Natural language processing (NLP) is an interdisciplinary subfield of computer science and information retrieval.
            It deals with the interaction between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.
        "#
    }
}
