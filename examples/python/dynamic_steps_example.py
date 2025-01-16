import cognee
import asyncio
import logging

from cognee.api.v1.search import SearchType
from cognee.shared.utils import setup_logging

job_1 = """
CV 1: Relevant
Name: Dr. Emily Carter
Contact Information:

Email: emily.carter@example.com
Phone: (555) 123-4567
Summary:

Senior Data Scientist with over 8 years of experience in machine learning and predictive analytics. Expertise in developing advanced algorithms and deploying scalable models in production environments.

Education:

Ph.D. in Computer Science, Stanford University (2014)
B.S. in Mathematics, University of California, Berkeley (2010)
Experience:

Senior Data Scientist, InnovateAI Labs (2016 – Present)
Led a team in developing machine learning models for natural language processing applications.
Implemented deep learning algorithms that improved prediction accuracy by 25%.
Collaborated with cross-functional teams to integrate models into cloud-based platforms.
Data Scientist, DataWave Analytics (2014 – 2016)
Developed predictive models for customer segmentation and churn analysis.
Analyzed large datasets using Hadoop and Spark frameworks.
Skills:

Programming Languages: Python, R, SQL
Machine Learning: TensorFlow, Keras, Scikit-Learn
Big Data Technologies: Hadoop, Spark
Data Visualization: Tableau, Matplotlib
"""

job_2 = """
CV 2: Relevant
Name: Michael Rodriguez
Contact Information:

Email: michael.rodriguez@example.com
Phone: (555) 234-5678
Summary:

Data Scientist with a strong background in machine learning and statistical modeling. Skilled in handling large datasets and translating data into actionable business insights.

Education:

M.S. in Data Science, Carnegie Mellon University (2013)
B.S. in Computer Science, University of Michigan (2011)
Experience:

Senior Data Scientist, Alpha Analytics (2017 – Present)
Developed machine learning models to optimize marketing strategies.
Reduced customer acquisition cost by 15% through predictive modeling.
Data Scientist, TechInsights (2013 – 2017)
Analyzed user behavior data to improve product features.
Implemented A/B testing frameworks to evaluate product changes.
Skills:

Programming Languages: Python, Java, SQL
Machine Learning: Scikit-Learn, XGBoost
Data Visualization: Seaborn, Plotly
Databases: MySQL, MongoDB
"""


job_3 = """
CV 3: Relevant
Name: Sarah Nguyen
Contact Information:

Email: sarah.nguyen@example.com
Phone: (555) 345-6789
Summary:

Data Scientist specializing in machine learning with 6 years of experience. Passionate about leveraging data to drive business solutions and improve product performance.

Education:

M.S. in Statistics, University of Washington (2014)
B.S. in Applied Mathematics, University of Texas at Austin (2012)
Experience:

Data Scientist, QuantumTech (2016 – Present)
Designed and implemented machine learning algorithms for financial forecasting.
Improved model efficiency by 20% through algorithm optimization.
Junior Data Scientist, DataCore Solutions (2014 – 2016)
Assisted in developing predictive models for supply chain optimization.
Conducted data cleaning and preprocessing on large datasets.
Skills:

Programming Languages: Python, R
Machine Learning Frameworks: PyTorch, Scikit-Learn
Statistical Analysis: SAS, SPSS
Cloud Platforms: AWS, Azure
"""


job_4 = """
CV 4: Not Relevant
Name: David Thompson
Contact Information:

Email: david.thompson@example.com
Phone: (555) 456-7890
Summary:

Creative Graphic Designer with over 8 years of experience in visual design and branding. Proficient in Adobe Creative Suite and passionate about creating compelling visuals.

Education:

B.F.A. in Graphic Design, Rhode Island School of Design (2012)
Experience:

Senior Graphic Designer, CreativeWorks Agency (2015 – Present)
Led design projects for clients in various industries.
Created branding materials that increased client engagement by 30%.
Graphic Designer, Visual Innovations (2012 – 2015)
Designed marketing collateral, including brochures, logos, and websites.
Collaborated with the marketing team to develop cohesive brand strategies.
Skills:

Design Software: Adobe Photoshop, Illustrator, InDesign
Web Design: HTML, CSS
Specialties: Branding and Identity, Typography
"""


job_5 = """
CV 5: Not Relevant
Name: Jessica Miller
Contact Information:

Email: jessica.miller@example.com
Phone: (555) 567-8901
Summary:

Experienced Sales Manager with a strong track record in driving sales growth and building high-performing teams. Excellent communication and leadership skills.

Education:

B.A. in Business Administration, University of Southern California (2010)
Experience:

Sales Manager, Global Enterprises (2015 – Present)
Managed a sales team of 15 members, achieving a 20% increase in annual revenue.
Developed sales strategies that expanded customer base by 25%.
Sales Representative, Market Leaders Inc. (2010 – 2015)
Consistently exceeded sales targets and received the 'Top Salesperson' award in 2013.
Skills:

Sales Strategy and Planning
Team Leadership and Development
CRM Software: Salesforce, Zoho
Negotiation and Relationship Building
"""


async def main(enable_steps):
    # Step 1: Reset data and system state
    if enable_steps.get("prune_data"):
        await cognee.prune.prune_data()
        print("Data pruned.")

    if enable_steps.get("prune_system"):
        await cognee.prune.prune_system(metadata=True)
        print("System pruned.")

    # Step 2: Add text
    if enable_steps.get("add_text"):
        text_list = [job_1, job_2, job_3, job_4, job_5]
        for text in text_list:
            await cognee.add(text)
            print(f"Added text: {text[:35]}...")

    # Step 3: Create knowledge graph
    if enable_steps.get("cognify"):
        await cognee.cognify()
        print("Knowledge graph created.")

    # Step 4: Query insights
    if enable_steps.get("retriever"):
        search_results = await cognee.search(
            SearchType.GRAPH_COMPLETION, query_text="Who has experience in design tools?"
        )
        print(search_results)


if __name__ == "__main__":
    setup_logging(logging.ERROR)

    rebuild_kg = True
    retrieve = True
    steps_to_enable = {
        "prune_data": rebuild_kg,
        "prune_system": rebuild_kg,
        "add_text": rebuild_kg,
        "cognify": rebuild_kg,
        "retriever": retrieve,
    }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(steps_to_enable))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
