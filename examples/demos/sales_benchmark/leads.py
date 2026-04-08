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
            "hitting walls. It works for one group but breaks down when we try to "
            "go broader. What does Cognee offer for growing AI platforms?"
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
            "Our AI works great in the lab but we're nervous about going to production "
            "with real customers. There are things we need to figure out first. "
            "How can Cognee help us get production-ready?"
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
            "We need to expand our AI to new business units but keep running into "
            "blockers from leadership. They say we're not ready. We need a platform "
            "that satisfies both engineering and the business side. Is Cognee the right fit?"
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
            "but they keep raising concerns during due diligence. We need to mature "
            "our platform. Can Cognee help us become enterprise-ready?"
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
            "Our AI outputs are unreliable and we think it's a data problem. "
            "The information going in is messy and the results coming out reflect that. "
            "What can Cognee do to improve our AI's output quality?"
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
            "We have tons of data but our AI can't make sense of it. "
            "There's valuable information buried in there but the AI misses "
            "important connections. How does Cognee approach this?"
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
            "Our AI doesn't understand our domain well enough. It gives generic "
            "answers when we need domain-specific intelligence. "
            "Is Cognee the right tool to make our AI smarter about our data?"
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
            "We process thousands of documents daily but our AI treats them all "
            "the same. We're not getting the intelligence we expected from this data. "
            "Can Cognee help us get more value from our documents?"
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
            "Our AI isn't performing well enough. Users interact with it daily "
            "but it doesn't seem to get any smarter. We need it to improve over time. "
            "How can Cognee help?"
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
            "Our AI product feels static — it gives the same quality answers whether "
            "it's day one or day hundred. We expected it to get better with use. "
            "Is Cognee relevant for making AI that improves?"
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
            "Our AI keeps making the same kinds of mistakes over and over. "
            "It doesn't seem to learn from its errors. We need something that "
            "actually gets better. Can Cognee handle this?"
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
            "Our AI assistant plateaued in quality months ago. Users interact with it "
            "constantly but it never improves. We're looking for a way to close that loop. "
            "What does Cognee offer?"
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
            "Our AI tools are underperforming. The team uses them daily but "
            "the experience feels disconnected — like starting from scratch every time. "
            "How does Cognee make AI more effective for teams?"
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
            "Our AI doesn't feel intelligent. It can answer questions but it doesn't "
            "seem to understand the bigger picture of what each person is working on. "
            "Can Cognee make our AI feel smarter?"
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
            "Our team keeps saying the AI isn't useful because it lacks context. "
            "They want something that understands their work, not just answers questions. "
            "How can Cognee help make AI more contextual?"
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
            "We invested in AI tools but they feel generic. Every interaction is "
            "stateless — no awareness of ongoing work or priorities. "
            "Can Cognee add that layer of intelligence?"
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
            "Our team's productivity is suffering. Everyone has information the others "
            "need but sharing it is painful. We're losing institutional knowledge. "
            "How can Cognee help with this?"
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
            "We have a knowledge management problem. Important information exists "
            "but people can't find it when they need it. Our AI doesn't help because "
            "it only sees a fraction of our data. Can Cognee solve this?"
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
            "Our organization has grown but our knowledge hasn't kept up. New people "
            "can't find what they need and veterans hoard information. We need AI to "
            "help. Is Cognee the right tool?"
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
            "Our AI assistant only knows about a small slice of our business data. "
            "There's so much more it could use but we can't figure out how to connect "
            "everything. How can Cognee help us unlock our data?"
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
            "Our AI gets things wrong too often. We've tried improving prompts and "
            "fine-tuning but accuracy is still not where it needs to be. "
            "What does Cognee offer that could help?"
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
            "We don't trust our AI's outputs enough to put them in front of customers. "
            "The quality is inconsistent and we can't figure out why. "
            "How does Cognee help improve AI quality?"
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
            "Our AI confidently gives wrong answers and our users are frustrated. "
            "We've thrown more data at it but that hasn't helped. "
            "How can Cognee make our AI more dependable?"
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
            "Our AI makes costly mistakes in production. We need to dramatically "
            "improve its accuracy but aren't sure what's missing. "
            "What's Cognee's approach to making AI more accurate?"
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
            "We're growing fast and our AI platform needs to keep up. We're adding "
            "customers quickly but things start breaking at scale. "
            "What's Cognee's approach to enterprise AI?"
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
            "Our sales team keeps losing deals at the security review stage. "
            "Prospects love the product but something in our architecture gives them "
            "pause. Can Cognee help us close these gaps?"
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
            "We work on sensitive projects and need AI infrastructure that meets "
            "strict requirements. Our current setup isn't cutting it. "
            "How does Cognee handle high-stakes deployments?"
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
            "We're in a regulated industry and need AI that works within strict "
            "constraints. We've been burned by platforms that can't meet our requirements. "
            "Is Cognee built for regulated environments?"
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
            "Our AI gives shallow answers because it doesn't really understand our "
            "domain. We have all the data but the AI treats it like generic text. "
            "What can Cognee do to make AI understand our field?"
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
            "We feed our AI lots of data but the outputs are disappointing. "
            "It's like the AI can't see the forest for the trees. "
            "How does Cognee help AI work better with complex data?"
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
            "We have massive amounts of content but our AI doesn't leverage it well. "
            "The answers it gives don't reflect the depth of knowledge we have. "
            "Is Cognee the right approach to improve this?"
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
            "Our AI can't answer domain-specific questions accurately. We have the "
            "data but something is lost in translation. "
            "Can Cognee help our AI truly understand our business?"
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
            "Our AI product isn't sticky enough — users try it and churn. "
            "It feels the same every time they use it. We need it to feel alive. "
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
            "We deployed AI six months ago and the quality hasn't budged. "
            "It's not getting worse but it's not getting better either. "
            "Can Cognee help us break through this plateau?"
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
            "We have tons of user interaction data but our AI ignores it. "
            "Users engage with it daily and we capture everything, but the AI "
            "doesn't benefit. What can Cognee do here?"
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
            "Our AI keeps making the same bad recommendations. Users tell us what "
            "they want but the AI doesn't adapt. We need a smarter system. "
            "How does Cognee make AI better over time?"
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
            "Our AI tools feel dumb. People use them for complex work but the AI "
            "treats every interaction as an isolated event. We need more intelligence. "
            "How can Cognee help?"
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
            "Our team complains the AI wastes their time. They spend more effort "
            "prompting it than they get back in value. Something fundamental is missing. "
            "Is Cognee the answer?"
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
            "We've built AI features but adoption is low. Users say it's not worth "
            "the effort because it doesn't understand their work. "
            "How does Cognee make AI more useful for knowledge workers?"
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
            "Our AI gives everyone the same experience regardless of their expertise "
            "or what they're working on. Power users want more. "
            "Can Cognee make AI adapt to individual users?"
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
            "Our AI only knows about a fraction of what our company knows. "
            "There's a massive gap between what we have and what the AI can access. "
            "What does Cognee offer to close that gap?"
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
            "We have rich company knowledge but our AI can't tap into most of it. "
            "People still go to colleagues instead of the AI because it doesn't "
            "know enough. Can Cognee fix this?"
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
            "Onboarding new people takes forever because knowledge is hard to find. "
            "The AI should help but it doesn't know about most of our work. "
            "How can Cognee help with organizational knowledge?"
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
            "Our team wastes hours hunting for information that exists somewhere "
            "in the company. The AI doesn't help because it can't see everything. "
            "Can Cognee make our knowledge accessible?"
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
            "Our AI is a liability right now. It produces plausible-sounding outputs "
            "that turn out to be wrong. We need to fix this before someone gets hurt. "
            "How does Cognee help?"
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
            "Our AI's error rate is too high for our industry. We've tried prompt "
            "engineering and fine-tuning but accuracy plateaued. "
            "How can Cognee help us get to production-grade accuracy?"
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
            "We can't deploy our AI because it's not accurate enough. "
            "The potential is there but we need a step-change in quality. "
            "What's Cognee's approach to improving AI outputs?"
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
            "Missing information in our AI's answers costs us real money. "
            "It's not that the AI is bad — it just doesn't find everything it should. "
            "How does Cognee approach this problem?"
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

TARGET_LEADS = 198  # 33 per archetype (divisible by 6)
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
