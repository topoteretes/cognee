"""Customer lead definitions for the sales benchmark."""

from __future__ import annotations

from pydantic import BaseModel


class BuyingProfile(BaseModel):
    """Hidden buying profile for a simulated customer lead."""

    lead_id: str
    persona_tag: str  # Hidden from sales agent — only used for metrics/traces
    company_context: str
    must_have_feature: str
    preferred_framing: str
    objection_style: str
    deal_breaker: str
    close_trigger: str
    hint_theme: str  # What the customer hints at when pitched wrong feature
    initial_message: str  # Deliberately ambiguous — doesn't reveal the needed feature


_ALL_LEADS = [
    # --- Archetype A (x4): talks about "scaling AI" → actually needs ACCESS CONTROL ---
    # The misdirection: "scaling" sounds like it needs infrastructure/retrieval,
    # but these customers' real blocker is data governance.
    BuyingProfile(
        lead_id="L01",
        persona_tag="scaling_ai",
        company_context="Series A startup, 12 engineers",
        must_have_feature="access_control",
        preferred_framing="compliance",
        objection_style="needs_proof",
        deal_breaker="no_tenant_isolation_story",
        close_trigger="compliance_case_study",
        hint_theme="the real issue is that different teams need isolated data environments",
        initial_message=(
            "We're trying to scale our AI system across the organization but keep "
            "hitting walls. Every department wants to use it but we can't just give "
            "everyone access to everything. What does Cognee offer?"
        ),
    ),
    BuyingProfile(
        lead_id="L02",
        persona_tag="scaling_ai",
        company_context="Seed-stage startup, 5 engineers",
        must_have_feature="access_control",
        preferred_framing="compliance",
        objection_style="integration_worry",
        deal_breaker="no_audit_trail",
        close_trigger="compliance_case_study",
        hint_theme="we need boundaries between user groups before we can roll out further",
        initial_message=(
            "Our AI platform works great for one team but we're nervous about rolling "
            "it out company-wide. There are concerns about who sees what. "
            "How can Cognee help us scale safely?"
        ),
    ),
    BuyingProfile(
        lead_id="L03",
        persona_tag="scaling_ai",
        company_context="Series B startup, 30 engineers",
        must_have_feature="access_control",
        preferred_framing="compliance",
        objection_style="scope_concern",
        deal_breaker="no_tenant_isolation_story",
        close_trigger="compliance_case_study",
        hint_theme="compliance and data separation are blocking our expansion",
        initial_message=(
            "We need to expand our AI to new business units but legal keeps blocking "
            "the rollout. Something about data boundaries. We need a platform that "
            "satisfies both engineering and legal. Is Cognee the right fit?"
        ),
    ),
    BuyingProfile(
        lead_id="L04",
        persona_tag="scaling_ai",
        company_context="Pre-seed startup, 3 engineers",
        must_have_feature="access_control",
        preferred_framing="simplicity",
        objection_style="too_expensive",
        deal_breaker="no_audit_trail",
        close_trigger="compliance_case_study",
        hint_theme="we need simple data isolation before our enterprise customers will sign",
        initial_message=(
            "We're a small team trying to sell our AI product to enterprise customers "
            "but they keep asking about data isolation. We don't have a good answer. "
            "Can Cognee help us check that box?"
        ),
    ),
    # --- Archetype B (x4): talks about "data quality" → actually needs KNOWLEDGE STRUCTURING ---
    # The misdirection: "data quality" sounds like feedback/monitoring,
    # but these customers need to impose structure on messy data.
    BuyingProfile(
        lead_id="L05",
        persona_tag="data_quality",
        company_context="Solo developer, data analytics tool",
        must_have_feature="knowledge_structuring",
        preferred_framing="developer_experience",
        objection_style="needs_proof",
        deal_breaker="no_custom_schema_support",
        close_trigger="custom_ontology_demo",
        hint_theme="we need to define relationships and structure in our messy data",
        initial_message=(
            "Our data is a mess — inconsistent formats, no clear relationships between "
            "records, duplicates everywhere. We need to clean it up and make it useful. "
            "What can Cognee do about data quality?"
        ),
    ),
    BuyingProfile(
        lead_id="L06",
        persona_tag="data_quality",
        company_context="Indie developer, building CRM tool",
        must_have_feature="knowledge_structuring",
        preferred_framing="developer_experience",
        objection_style="too_abstract",
        deal_breaker="cant_define_own_entity_types",
        close_trigger="custom_ontology_demo",
        hint_theme="we need to extract entities and map connections from unstructured text",
        initial_message=(
            "We have tons of unstructured data but no way to make sense of it. "
            "There's valuable information buried in there but we can't see the patterns. "
            "How does Cognee approach data quality problems?"
        ),
    ),
    BuyingProfile(
        lead_id="L07",
        persona_tag="data_quality",
        company_context="Small consulting firm",
        must_have_feature="knowledge_structuring",
        preferred_framing="simplicity",
        objection_style="scope_concern",
        deal_breaker="no_custom_schema_support",
        close_trigger="custom_ontology_demo",
        hint_theme="turning flat unstructured data into a connected graph with custom types",
        initial_message=(
            "Our client data exists as free-text reports and scattered notes. Nothing "
            "is structured. We want to turn it into something organized and queryable. "
            "Is Cognee the right tool for this?"
        ),
    ),
    BuyingProfile(
        lead_id="L08",
        persona_tag="data_quality",
        company_context="Small team, 4 engineers",
        must_have_feature="knowledge_structuring",
        preferred_framing="developer_experience",
        objection_style="integration_worry",
        deal_breaker="cant_define_own_entity_types",
        close_trigger="custom_ontology_demo",
        hint_theme="we need domain-specific entity extraction with our own schema definitions",
        initial_message=(
            "We process thousands of documents daily but the data comes in messy. "
            "We need to extract the important bits and connect them systematically. "
            "Can Cognee help with our data quality pipeline?"
        ),
    ),
    # --- Archetype C (x4): talks about "AI personalization" → actually needs FEEDBACK ---
    # The misdirection: "personalization" sounds like it needs memory,
    # but these customers need the AI to learn from corrections.
    BuyingProfile(
        lead_id="L09",
        persona_tag="ai_personalization",
        company_context="E-commerce company, recommendations team",
        must_have_feature="feedback",
        preferred_framing="roi",
        objection_style="needs_proof",
        deal_breaker="no_concrete_demo_of_self_improvement",
        close_trigger="self_improvement_loop_demo",
        hint_theme="the AI needs to learn from user corrections and get better over time",
        initial_message=(
            "We want our AI to feel personalized — it should adapt to each user's "
            "preferences over time. Right now it gives generic responses that users "
            "keep correcting. How can Cognee help with personalization?"
        ),
    ),
    BuyingProfile(
        lead_id="L10",
        persona_tag="ai_personalization",
        company_context="EdTech platform",
        must_have_feature="feedback",
        preferred_framing="developer_experience",
        objection_style="too_abstract",
        deal_breaker="requires_major_architecture_changes",
        close_trigger="self_improvement_loop_demo",
        hint_theme="the system should improve its outputs based on explicit user feedback",
        initial_message=(
            "Our AI tutor needs to personalize to each student. Students tell it when "
            "explanations don't make sense but it never learns. We want it to adapt. "
            "Is Cognee relevant for AI personalization?"
        ),
    ),
    BuyingProfile(
        lead_id="L11",
        persona_tag="ai_personalization",
        company_context="Content platform",
        must_have_feature="feedback",
        preferred_framing="simplicity",
        objection_style="scope_concern",
        deal_breaker="no_concrete_demo_of_self_improvement",
        close_trigger="self_improvement_loop_demo",
        hint_theme="we need the AI to stop repeating mistakes that users flag",
        initial_message=(
            "Our content AI keeps making the same mistakes even though editors flag "
            "them every time. We need it to feel more personalized — to learn from "
            "what people tell it. Can Cognee handle this?"
        ),
    ),
    BuyingProfile(
        lead_id="L12",
        persona_tag="ai_personalization",
        company_context="Customer success team, SaaS",
        must_have_feature="feedback",
        preferred_framing="roi",
        objection_style="too_expensive",
        deal_breaker="requires_major_architecture_changes",
        close_trigger="self_improvement_loop_demo",
        hint_theme="the AI should incorporate corrections and not repeat flagged mistakes",
        initial_message=(
            "We want our AI assistant to feel personal — like it knows each customer's "
            "history and preferences. Customers keep telling it things but it forgets. "
            "What does Cognee offer for personalized AI experiences?"
        ),
    ),
    # --- Archetype D (x4): talks about "search improvement" → actually needs MEMORY ---
    # The misdirection: "search" sounds like retrieval,
    # but these customers need persistent context that builds over time.
    BuyingProfile(
        lead_id="L13",
        persona_tag="search_improvement",
        company_context="Research lab, biotech",
        must_have_feature="memory",
        preferred_framing="research",
        objection_style="needs_proof",
        deal_breaker="no_per_user_isolation",
        close_trigger="persistent_context_demo",
        hint_theme="each user needs the AI to build up context over time from past sessions",
        initial_message=(
            "Our team keeps searching for the same things over and over. The AI "
            "doesn't remember what we looked for last week. We need smarter search "
            "that learns from our patterns. How does Cognee improve search?"
        ),
    ),
    BuyingProfile(
        lead_id="L14",
        persona_tag="search_improvement",
        company_context="Law firm, case research",
        must_have_feature="memory",
        preferred_framing="simplicity",
        objection_style="scope_concern",
        deal_breaker="too_complex_to_set_up",
        close_trigger="persistent_context_demo",
        hint_theme="the system should accumulate knowledge from each user's research history",
        initial_message=(
            "Our lawyers search our case database constantly but the AI treats every "
            "search as if it's the first time. It should know what each attorney is "
            "working on. Can Cognee make our search more context-aware?"
        ),
    ),
    BuyingProfile(
        lead_id="L15",
        persona_tag="search_improvement",
        company_context="Consulting firm, knowledge management",
        must_have_feature="memory",
        preferred_framing="simplicity",
        objection_style="too_abstract",
        deal_breaker="no_per_user_isolation",
        close_trigger="persistent_context_demo",
        hint_theme="per-user context that persists and grows across sessions",
        initial_message=(
            "Our consultants waste time re-explaining context to our AI tools every "
            "session. The system should know their project, their clients, their history. "
            "How can Cognee help our search understand user context?"
        ),
    ),
    BuyingProfile(
        lead_id="L16",
        persona_tag="search_improvement",
        company_context="Venture capital firm",
        must_have_feature="memory",
        preferred_framing="developer_experience",
        objection_style="integration_worry",
        deal_breaker="too_complex_to_set_up",
        close_trigger="persistent_context_demo",
        hint_theme="accumulated research context per analyst that informs future searches",
        initial_message=(
            "Our analysts research companies and the AI should learn from their past "
            "searches and notes. Right now search results are generic — no sense of "
            "what each person cares about. Can Cognee add that intelligence?"
        ),
    ),
    # --- Archetype E (x4): talks about "team collaboration" → actually needs MULTIMODAL INGESTION ---
    # The misdirection: "collaboration" sounds like access control or memory,
    # but these customers need to unify scattered data sources.
    BuyingProfile(
        lead_id="L17",
        persona_tag="team_collaboration",
        company_context="Marketing agency",
        must_have_feature="multimodal_ingestion",
        preferred_framing="simplicity",
        objection_style="too_abstract",
        deal_breaker="requires_coding_to_set_up",
        close_trigger="no_code_setup",
        hint_theme="we need to pull in data from many different tools and formats into one place",
        initial_message=(
            "Our team can't collaborate effectively because everyone's knowledge lives "
            "in different tools. Design in Figma, copy in Docs, feedback in Slack, "
            "analytics in dashboards. How can Cognee help us work together better?"
        ),
    ),
    BuyingProfile(
        lead_id="L18",
        persona_tag="team_collaboration",
        company_context="Product team, mid-size company",
        must_have_feature="multimodal_ingestion",
        preferred_framing="roi",
        objection_style="too_expensive",
        deal_breaker="no_clear_roi_story",
        close_trigger="no_code_setup",
        hint_theme="unifying information from scattered systems into one searchable layer",
        initial_message=(
            "Cross-team collaboration is our biggest pain point. Engineers use GitHub, "
            "PMs use Notion, support uses Zendesk. Nobody sees the full picture. "
            "Can Cognee help break down these silos?"
        ),
    ),
    BuyingProfile(
        lead_id="L19",
        persona_tag="team_collaboration",
        company_context="Non-profit, distributed team",
        must_have_feature="multimodal_ingestion",
        preferred_framing="simplicity",
        objection_style="scope_concern",
        deal_breaker="requires_coding_to_set_up",
        close_trigger="no_code_setup",
        hint_theme="pulling together documents, emails, and notes from different formats",
        initial_message=(
            "We're a distributed team and our knowledge is scattered across email, "
            "shared drives, and chat. New team members can't find anything. We need "
            "a way to bring it all together. Is Cognee the right tool?"
        ),
    ),
    BuyingProfile(
        lead_id="L20",
        persona_tag="team_collaboration",
        company_context="Real estate platform",
        must_have_feature="multimodal_ingestion",
        preferred_framing="roi",
        objection_style="too_expensive",
        deal_breaker="no_clear_roi_story",
        close_trigger="no_code_setup",
        hint_theme="combining structured data, images, and text from multiple systems",
        initial_message=(
            "Our team collaboration suffers because data is in different formats and "
            "systems. Listings, photos, reviews, agent notes — all disconnected. "
            "How can Cognee help our team access everything in one place?"
        ),
    ),
    # --- Archetype F (x4): talks about "AI reliability" → actually needs RETRIEVAL ---
    # The misdirection: "reliability" sounds like feedback/monitoring,
    # but these customers need better retrieval to ground AI responses in facts.
    BuyingProfile(
        lead_id="L21",
        persona_tag="ai_reliability",
        company_context="Healthcare AI company",
        must_have_feature="retrieval",
        preferred_framing="research",
        objection_style="needs_proof",
        deal_breaker="no_advanced_search",
        close_trigger="graph_traversal_demo",
        hint_theme="the AI needs to find and cite the right source material before answering",
        initial_message=(
            "Our AI gives confident but sometimes wrong answers. We need it to be "
            "more reliable — grounded in actual data, not hallucinations. "
            "What does Cognee offer for AI reliability?"
        ),
    ),
    BuyingProfile(
        lead_id="L22",
        persona_tag="ai_reliability",
        company_context="Financial services, compliance reporting",
        must_have_feature="retrieval",
        preferred_framing="compliance",
        objection_style="scope_concern",
        deal_breaker="no_advanced_search",
        close_trigger="graph_traversal_demo",
        hint_theme="finding the right information to back up AI responses with citations",
        initial_message=(
            "We can't trust our AI's outputs because we can't verify where it gets "
            "its information. We need reliable, traceable AI responses. "
            "How does Cognee help with AI trustworthiness?"
        ),
    ),
    BuyingProfile(
        lead_id="L23",
        persona_tag="ai_reliability",
        company_context="Legal tech startup",
        must_have_feature="retrieval",
        preferred_framing="research",
        objection_style="too_abstract",
        deal_breaker="no_advanced_search",
        close_trigger="graph_traversal_demo",
        hint_theme="semantic and graph-based search that finds contextually relevant information",
        initial_message=(
            "Our AI sometimes makes things up and our users have lost trust. We need "
            "it to always pull from verified sources. How can Cognee make our AI "
            "more dependable?"
        ),
    ),
    BuyingProfile(
        lead_id="L24",
        persona_tag="ai_reliability",
        company_context="Insurance company, claims processing",
        must_have_feature="retrieval",
        preferred_framing="roi",
        objection_style="too_expensive",
        deal_breaker="no_advanced_search",
        close_trigger="graph_traversal_demo",
        hint_theme="traversing relationships in data to find accurate, relevant answers",
        initial_message=(
            "Our AI processes claims but makes errors because it can't find the right "
            "policy documents. We need it to be more accurate. What does Cognee's "
            "approach to reliability look like?"
        ),
    ),
    # --- Archetype A continued (x4 more): "scaling AI" → ACCESS CONTROL ---
    BuyingProfile(
        lead_id="L25",
        persona_tag="scaling_ai",
        company_context="Growth-stage fintech, 20 engineers",
        must_have_feature="access_control",
        preferred_framing="compliance",
        objection_style="needs_proof",
        deal_breaker="no_tenant_isolation_story",
        close_trigger="compliance_case_study",
        hint_theme="different customer segments need strictly separated data environments",
        initial_message=(
            "We're onboarding enterprise clients onto our AI platform but each client "
            "insists their data stays separate. We need to scale without mixing anything up. "
            "What's Cognee's approach?"
        ),
    ),
    BuyingProfile(
        lead_id="L26",
        persona_tag="scaling_ai",
        company_context="B2B SaaS, AI analytics product",
        must_have_feature="access_control",
        preferred_framing="simplicity",
        objection_style="integration_worry",
        deal_breaker="no_audit_trail",
        close_trigger="compliance_case_study",
        hint_theme="multi-tenant data separation is the real blocker for growth",
        initial_message=(
            "Every new customer we sign asks 'can you guarantee our data won't leak to "
            "other tenants?' and we don't have a great answer yet. We need to solve this "
            "to scale. Can Cognee help?"
        ),
    ),
    BuyingProfile(
        lead_id="L27",
        persona_tag="scaling_ai",
        company_context="Government contractor, AI division",
        must_have_feature="access_control",
        preferred_framing="compliance",
        objection_style="scope_concern",
        deal_breaker="no_tenant_isolation_story",
        close_trigger="compliance_case_study",
        hint_theme="strict compartmentalization requirements between projects and clearance levels",
        initial_message=(
            "We're deploying AI across multiple government projects but each has different "
            "security clearances. We can't have cross-contamination. How does Cognee "
            "handle these kinds of scaling challenges?"
        ),
    ),
    BuyingProfile(
        lead_id="L28",
        persona_tag="scaling_ai",
        company_context="Healthcare startup, HIPAA-regulated",
        must_have_feature="access_control",
        preferred_framing="compliance",
        objection_style="too_expensive",
        deal_breaker="no_audit_trail",
        close_trigger="compliance_case_study",
        hint_theme="patient data must be strictly isolated per provider organization",
        initial_message=(
            "We want to offer our AI diagnostic tool to multiple hospital networks but "
            "patient data privacy is non-negotiable. How do we scale this responsibly? "
            "Is Cognee built for this?"
        ),
    ),
    # --- Archetype B continued (x4 more): "data quality" → KNOWLEDGE STRUCTURING ---
    BuyingProfile(
        lead_id="L29",
        persona_tag="data_quality",
        company_context="Biotech research lab",
        must_have_feature="knowledge_structuring",
        preferred_framing="research",
        objection_style="needs_proof",
        deal_breaker="no_custom_schema_support",
        close_trigger="custom_ontology_demo",
        hint_theme="turning raw experimental data into a structured knowledge base with custom entities",
        initial_message=(
            "Our research data is scattered across PDFs, spreadsheets, and lab notebooks. "
            "Nothing connects to anything else. We need better data quality to make "
            "discoveries. What can Cognee do?"
        ),
    ),
    BuyingProfile(
        lead_id="L30",
        persona_tag="data_quality",
        company_context="Supply chain analytics startup",
        must_have_feature="knowledge_structuring",
        preferred_framing="simplicity",
        objection_style="too_abstract",
        deal_breaker="cant_define_own_entity_types",
        close_trigger="custom_ontology_demo",
        hint_theme="defining custom entity types and relationships from messy operational data",
        initial_message=(
            "We get data from hundreds of suppliers in different formats. It's a data "
            "quality nightmare. We need to normalize and connect it all. "
            "How does Cognee handle messy data?"
        ),
    ),
    BuyingProfile(
        lead_id="L31",
        persona_tag="data_quality",
        company_context="Academic publishing platform",
        must_have_feature="knowledge_structuring",
        preferred_framing="developer_experience",
        objection_style="scope_concern",
        deal_breaker="no_custom_schema_support",
        close_trigger="custom_ontology_demo",
        hint_theme="extracting structured metadata and citation graphs from unstructured papers",
        initial_message=(
            "We have millions of academic papers but the metadata is inconsistent. "
            "Authors, citations, topics — all messy. We need to clean this up at scale. "
            "Is Cognee the right approach for data quality?"
        ),
    ),
    BuyingProfile(
        lead_id="L32",
        persona_tag="data_quality",
        company_context="Manufacturing company, quality control",
        must_have_feature="knowledge_structuring",
        preferred_framing="roi",
        objection_style="integration_worry",
        deal_breaker="cant_define_own_entity_types",
        close_trigger="custom_ontology_demo",
        hint_theme="building a structured graph from free-text quality reports and inspection logs",
        initial_message=(
            "Our quality reports are all free text — inspectors write whatever they want. "
            "We can't analyze trends because the data is unstructured garbage. "
            "Can Cognee fix our data quality issues?"
        ),
    ),
    # --- Archetype C continued (x4 more): "AI personalization" → FEEDBACK ---
    BuyingProfile(
        lead_id="L33",
        persona_tag="ai_personalization",
        company_context="Fitness app, AI coaching",
        must_have_feature="feedback",
        preferred_framing="simplicity",
        objection_style="needs_proof",
        deal_breaker="no_concrete_demo_of_self_improvement",
        close_trigger="self_improvement_loop_demo",
        hint_theme="the AI coach needs to incorporate user corrections into future suggestions",
        initial_message=(
            "Our AI fitness coach gives the same generic advice to everyone. Users want "
            "it to learn their preferences and adapt. We need real personalization. "
            "How does Cognee approach this?"
        ),
    ),
    BuyingProfile(
        lead_id="L34",
        persona_tag="ai_personalization",
        company_context="HR tech, employee onboarding",
        must_have_feature="feedback",
        preferred_framing="roi",
        objection_style="too_abstract",
        deal_breaker="requires_major_architecture_changes",
        close_trigger="self_improvement_loop_demo",
        hint_theme="learning from manager corrections to improve onboarding recommendations",
        initial_message=(
            "Our AI onboarding assistant needs to be personalized per department. "
            "Managers keep correcting it but it never improves. We need adaptive AI. "
            "Can Cognee deliver personalized experiences?"
        ),
    ),
    BuyingProfile(
        lead_id="L35",
        persona_tag="ai_personalization",
        company_context="Music streaming platform",
        must_have_feature="feedback",
        preferred_framing="developer_experience",
        objection_style="scope_concern",
        deal_breaker="no_concrete_demo_of_self_improvement",
        close_trigger="self_improvement_loop_demo",
        hint_theme="using thumbs up/down signals to refine what the AI recommends next",
        initial_message=(
            "Users skip songs our AI recommends and we have all that signal data but "
            "the AI doesn't get better. We need personalization that actually learns. "
            "What can Cognee do here?"
        ),
    ),
    BuyingProfile(
        lead_id="L36",
        persona_tag="ai_personalization",
        company_context="Travel booking AI assistant",
        must_have_feature="feedback",
        preferred_framing="roi",
        objection_style="too_expensive",
        deal_breaker="requires_major_architecture_changes",
        close_trigger="self_improvement_loop_demo",
        hint_theme="incorporating booking outcomes and user complaints to improve future suggestions",
        initial_message=(
            "Our travel AI suggests trips but users keep saying 'no, not like that.' "
            "It should learn from rejections and get better over time. How does Cognee "
            "handle AI personalization?"
        ),
    ),
    # --- Archetype D continued (x4 more): "search improvement" → MEMORY ---
    BuyingProfile(
        lead_id="L37",
        persona_tag="search_improvement",
        company_context="Patent search firm",
        must_have_feature="memory",
        preferred_framing="research",
        objection_style="needs_proof",
        deal_breaker="no_per_user_isolation",
        close_trigger="persistent_context_demo",
        hint_theme="each patent analyst needs accumulated research context across search sessions",
        initial_message=(
            "Our patent analysts run the same searches repeatedly because the system "
            "forgets everything between sessions. We need search that builds on past work. "
            "How can Cognee improve our search?"
        ),
    ),
    BuyingProfile(
        lead_id="L38",
        persona_tag="search_improvement",
        company_context="Journalism platform, investigative team",
        must_have_feature="memory",
        preferred_framing="simplicity",
        objection_style="scope_concern",
        deal_breaker="too_complex_to_set_up",
        close_trigger="persistent_context_demo",
        hint_theme="journalists need search that remembers their ongoing investigation context",
        initial_message=(
            "Our journalists research stories over weeks but every search starts from "
            "scratch. The AI should know what story they're working on and find relevant "
            "connections. Is Cognee good for search improvement?"
        ),
    ),
    BuyingProfile(
        lead_id="L39",
        persona_tag="search_improvement",
        company_context="Competitive intelligence firm",
        must_have_feature="memory",
        preferred_framing="developer_experience",
        objection_style="too_abstract",
        deal_breaker="no_per_user_isolation",
        close_trigger="persistent_context_demo",
        hint_theme="persistent per-analyst context that makes each search smarter than the last",
        initial_message=(
            "We track competitors and our analysts keep re-explaining context to our AI "
            "search tool. It should already know what companies and markets each person "
            "follows. How does Cognee make search smarter?"
        ),
    ),
    BuyingProfile(
        lead_id="L40",
        persona_tag="search_improvement",
        company_context="Academic library, research support",
        must_have_feature="memory",
        preferred_framing="research",
        objection_style="integration_worry",
        deal_breaker="too_complex_to_set_up",
        close_trigger="persistent_context_demo",
        hint_theme="building up a research profile per user that informs future search results",
        initial_message=(
            "Our researchers use our search system daily but it never gets smarter. "
            "A PhD student studying climate change gets the same results as a freshman. "
            "Can Cognee add context-awareness to search?"
        ),
    ),
    # --- Archetype E continued (x4 more): "team collaboration" → MULTIMODAL INGESTION ---
    BuyingProfile(
        lead_id="L41",
        persona_tag="team_collaboration",
        company_context="Architecture firm, design team",
        must_have_feature="multimodal_ingestion",
        preferred_framing="simplicity",
        objection_style="too_abstract",
        deal_breaker="requires_coding_to_set_up",
        close_trigger="no_code_setup",
        hint_theme="pulling in CAD files, emails, meeting notes, and photos into one system",
        initial_message=(
            "Our architects, engineers, and project managers use completely different "
            "tools. Knowledge gets lost between teams. We need better collaboration. "
            "What does Cognee offer?"
        ),
    ),
    BuyingProfile(
        lead_id="L42",
        persona_tag="team_collaboration",
        company_context="Film production company",
        must_have_feature="multimodal_ingestion",
        preferred_framing="roi",
        objection_style="too_expensive",
        deal_breaker="no_clear_roi_story",
        close_trigger="no_code_setup",
        hint_theme="unifying scripts, storyboards, budget sheets, and video notes in one place",
        initial_message=(
            "Our production team has scripts in Final Draft, budgets in Excel, notes in "
            "Google Docs, and dailies on a server. Nobody can find anything. "
            "Can Cognee help our team collaborate?"
        ),
    ),
    BuyingProfile(
        lead_id="L43",
        persona_tag="team_collaboration",
        company_context="Remote-first startup, 50 people",
        must_have_feature="multimodal_ingestion",
        preferred_framing="simplicity",
        objection_style="scope_concern",
        deal_breaker="requires_coding_to_set_up",
        close_trigger="no_code_setup",
        hint_theme="ingesting data from Slack, Notion, Drive, and Jira into a unified layer",
        initial_message=(
            "We're fully remote and our institutional knowledge is split across a dozen "
            "tools. Onboarding takes forever because people can't find past decisions. "
            "How can Cognee improve team collaboration?"
        ),
    ),
    BuyingProfile(
        lead_id="L44",
        persona_tag="team_collaboration",
        company_context="Event management company",
        must_have_feature="multimodal_ingestion",
        preferred_framing="roi",
        objection_style="too_expensive",
        deal_breaker="no_clear_roi_story",
        close_trigger="no_code_setup",
        hint_theme="combining vendor contracts, floor plans, photos, and communications into one view",
        initial_message=(
            "Every event we run has data in different places — vendor emails, floor plans "
            "as PDFs, photos, contracts. Our team wastes hours hunting for information. "
            "Can Cognee make collaboration easier?"
        ),
    ),
    # --- Archetype F continued (x4 more): "AI reliability" → RETRIEVAL ---
    BuyingProfile(
        lead_id="L45",
        persona_tag="ai_reliability",
        company_context="Pharmaceutical company, drug safety",
        must_have_feature="retrieval",
        preferred_framing="compliance",
        objection_style="needs_proof",
        deal_breaker="no_advanced_search",
        close_trigger="graph_traversal_demo",
        hint_theme="finding and citing specific regulatory documents before generating any answer",
        initial_message=(
            "Our AI generates drug interaction reports but sometimes cites wrong studies. "
            "We need bulletproof reliability — wrong answers could harm patients. "
            "How does Cognee ensure AI accuracy?"
        ),
    ),
    BuyingProfile(
        lead_id="L46",
        persona_tag="ai_reliability",
        company_context="Tax preparation software",
        must_have_feature="retrieval",
        preferred_framing="roi",
        objection_style="scope_concern",
        deal_breaker="no_advanced_search",
        close_trigger="graph_traversal_demo",
        hint_theme="looking up the correct tax code sections before answering any question",
        initial_message=(
            "Our AI tax assistant gives wrong answers about 5% of the time. That's "
            "unacceptable in tax — one wrong answer can cost a client thousands. "
            "How can Cognee make our AI more reliable?"
        ),
    ),
    BuyingProfile(
        lead_id="L47",
        persona_tag="ai_reliability",
        company_context="Engineering firm, safety documentation",
        must_have_feature="retrieval",
        preferred_framing="research",
        objection_style="too_abstract",
        deal_breaker="no_advanced_search",
        close_trigger="graph_traversal_demo",
        hint_theme="multi-hop graph traversal to find relevant safety standards and precedents",
        initial_message=(
            "Our safety engineers need AI that's 100% reliable — it must reference actual "
            "standards documents, not make things up. Current tools hallucinate too much. "
            "What's Cognee's approach to AI reliability?"
        ),
    ),
    BuyingProfile(
        lead_id="L48",
        persona_tag="ai_reliability",
        company_context="E-discovery legal platform",
        must_have_feature="retrieval",
        preferred_framing="compliance",
        objection_style="integration_worry",
        deal_breaker="no_advanced_search",
        close_trigger="graph_traversal_demo",
        hint_theme="precise document retrieval with relationship-aware search across case files",
        initial_message=(
            "In e-discovery, our AI needs to find every relevant document — missing one "
            "can lose a case. We need reliability we can stake our reputation on. "
            "How does Cognee approach this?"
        ),
    ),
]

# Group leads by archetype for interleaving
_ARCHETYPES = [
    [lead for lead in _ALL_LEADS if lead.persona_tag == "scaling_ai"],
    [lead for lead in _ALL_LEADS if lead.persona_tag == "data_quality"],
    [lead for lead in _ALL_LEADS if lead.persona_tag == "ai_personalization"],
    [lead for lead in _ALL_LEADS if lead.persona_tag == "search_improvement"],
    [lead for lead in _ALL_LEADS if lead.persona_tag == "team_collaboration"],
    [lead for lead in _ALL_LEADS if lead.persona_tag == "ai_reliability"],
]

TARGET_LEADS = 300  # 50 per archetype for robust demo stats
_LEADS_PER_ARCHETYPE = TARGET_LEADS // len(_ARCHETYPES)  # 34 each


def _generate_leads() -> list[BuyingProfile]:
    """Generate TARGET_LEADS leads by cycling through existing ones with new IDs.

    Interleaved: one from each archetype per round, then repeat.
    This ensures the agent sees diverse personas and must recall across gaps.
    """
    # Expand each archetype to _LEADS_PER_ARCHETYPE by cycling
    expanded: list[list[BuyingProfile]] = []
    for group in _ARCHETYPES:
        archetype_leads = []
        for i in range(_LEADS_PER_ARCHETYPE):
            source = group[i % len(group)]
            new_id = f"L{len(expanded) * _LEADS_PER_ARCHETYPE + i + 1:03d}"
            archetype_leads.append(source.model_copy(update={"lead_id": new_id}))
        expanded.append(archetype_leads)

    # Interleave: round-robin across archetypes
    leads = []
    for round_idx in range(_LEADS_PER_ARCHETYPE):
        for arch_leads in expanded:
            leads.append(arch_leads[round_idx])
    return leads


LEADS: list[BuyingProfile] = _generate_leads()
