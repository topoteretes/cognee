import asyncio
from pprint import pprint

import cognee
from cognee.api.v1.search import SearchType
from cognee.modules.cognify.config import get_cognify_config
from cognee.shared.logging_utils import ERROR, setup_logging


GROUPED_DOCUMENTS = [
    """
    Candidate Ada Mercer is a senior machine learning engineer at Nova Robotics.
    She led Project Sparrow, building Jetson Orin computer-vision pipelines for warehouse
    inspection robots. Her skills include Python, PyTorch, edge deployment, model evaluation,
    and latency optimization. Her measurable result was reducing defect-detection latency to
    14 milliseconds per frame while managing overheating risk.
    """,
    """
    Candidate Ben Ortiz is a reliability engineer at Kestrel Manufacturing.
    He owns predictive-maintenance models for assembly-line motors and works with vibration
    sensors, SQL, Grafana, and root-cause analysis. His certification is Six Sigma Green Belt,
    and his key metric is reducing unplanned downtime by 18 percent.
    """,
    """
    Candidate Claire Zhang is a healthcare operations analyst at Northbridge Hospital.
    She improved operating-room scheduling with MedQueue and SignalLake dashboards.
    Her skills include demand forecasting, stakeholder interviews, Python, Tableau, and
    staffing analytics. Her main project risk was adoption by surgical coordinators.
    """,
    """
    Candidate Daniel Cho is a radiology systems manager at Pinecrest Clinic.
    He coordinated MRI scheduling, machine maintenance windows, and radiologist coverage.
    His tools include ScanFlow Scheduler, Excel, SQL, and incident reporting. His success
    metric was reducing patient waitlists before the Q3 2026 rollout.
    """,
    """
    Candidate Elena Rossi is a platform engineer at Acme Cloud.
    She migrated backend releases from Jenkins to GitHub Actions and standardized reusable
    workflow templates. Her skills include Python, Docker, CI/CD, secrets rotation, and
    release governance. Her key metric was improving cache hit rate and reducing failed gates.
    """,
    """
    Candidate Farah Singh is a fraud risk data scientist at ArgentPay.
    She built account-takeover detection models using device fingerprints, login velocity,
    and behavioral biometrics. Her tools include scikit-learn, feature stores, SQL, and
    model monitoring. Her constraint was balancing blocked transfers against false positives.
    """,
    """
    Candidate Gabriel Reed is a supply-chain project manager at Verdant Foods.
    He managed molded-fiber packaging trials with GreenMold Supply and tracked shelf life,
    moisture barriers, lead time, and compostability certification. His budget responsibility
    was 280,000 dollars, and his timeline target was a Q4 packaging launch.
    """,
    """
    Candidate Hana Brooks is a field science technician at Alpine Research.
    She deployed snowpack sensors across mountain passes and monitored battery status,
    radio connectivity, calibration drift, and storm damage. Her tools include LoRa radios,
    sensor firmware, Python notebooks, and field maintenance logs.
    """,
    """
    Candidate Ivan Petrov is an API platform lead at Orbit API.
    He launched schema generation from FastAPI routes and managed SDK release checks.
    His skills include OpenAPI, Python, contract testing, developer documentation, and
    customer integration support. His key metric was reducing integration defects.
    """,
    """
    Candidate Maya Patel is a clinical transformation director at Lakeside Medical Center.
    She led staffing forecast work with ShiftWise Labs and VectorOps Consulting.
    Her responsibilities include nurse scheduling, patient acuity modeling, overtime reduction,
    governance board reporting, and adoption planning for charge nurses.
    """,
    """
    The Aster Motors Vela E4 is a compact electric hatchback designed for city commuting.
    It uses a 62 kWh battery pack, a single-motor drivetrain, adaptive cruise control,
    and lane-keeping assistance. The measurable performance target is 410 kilometers of range,
    while the main constraint is fast-charging availability on rural routes.
    """,
    """
    The Boreal Atlas X7 is a plug-in hybrid SUV built for family travel and winter roads.
    It includes all-wheel drive, a 2.0 liter turbo engine, battery preconditioning, roof-rail
    cargo support, and blind-spot monitoring. Its safety rating is five stars, and its towing
    capacity is 2,100 kilograms.
    """,
    """
    The Cobalt Sprint R is a performance coupe from Meridian Auto Works.
    It features a dual-clutch transmission, sport-tuned suspension, carbon-ceramic brakes,
    and telemetry dashboards. Its key metric is 0 to 100 km/h in 3.9 seconds, and the main
    ownership risk is high tire and brake replacement cost.
    """,
    """
    The Driftline CargoMax is a commercial electric van for delivery fleets.
    It offers modular shelving, fleet telematics, route optimization, regenerative braking,
    and 900 kilograms of payload capacity. The warranty covers eight years on the battery,
    and the launch window is October 2026.
    """,
    """
    The Everlight Terra S is a midsize sedan focused on comfort and driver assistance.
    It includes heated seats, acoustic glass, a 14-speaker audio system, automated parking,
    and over-the-air software updates. The manufacturer tracks customer satisfaction, cabin
    noise, service visits, and infotainment reliability.
    """,
    """
    The Falcon Ridge TrailPro is an off-road pickup designed for construction sites.
    It includes locking differentials, skid plates, hill-descent control, trailer sway control,
    and a reinforced suspension. The project budget for fleet conversion is 520,000 dollars,
    and the dependency is driver training approval.
    """,
    """
    The Glacier Mini EV is a low-cost city car from Northline Mobility.
    It has a 38 kWh battery, compact parking sensors, basic lane alerts, and recycled interior
    materials. Its metric is total cost of ownership, and the main risk is limited highway range.
    """,
    """
    The HelioTour Grand is a luxury electric crossover for executive transport.
    It uses air suspension, rear-seat climate zones, biometric driver profiles, and SignalLake
    fleet reporting. The service package includes concierge maintenance, and the timeline target
    is a Q3 2026 dealership rollout.
    """,
    """
    The IonWorks MetroCab is a purpose-built taxi model for dense urban fleets.
    It includes washable rear seating, high-cycle door hardware, driver safety partitions,
    route analytics, and rapid charging. Its key metric is uptime per vehicle, and the constraint
    is charging-station queue time during evening shifts.
    """,
    """
    The Juniper AeroWagon is a hybrid wagon for regional sales teams.
    It combines long cargo space, driver-assistance cameras from Helio Components, fuel-economy
    coaching, and maintenance dashboards reviewed by VectorOps Consulting. The fleet manager tracks
    service cost, driver adoption, route efficiency, and Q4 2026 readiness.
    """,
]


async def main(enable_steps):
    get_cognify_config().ontology_generation = "AUTO_LOW_LEVEL_CANONICAL"

    if enable_steps.get("prune_data"):
        await cognee.prune.prune_data()
        print("Data pruned.")

    if enable_steps.get("prune_system"):
        await cognee.prune.prune_system(metadata=True)
        print("System pruned.")

    if enable_steps.get("add_text"):
        print(f"Adding {len(GROUPED_DOCUMENTS)} grouped documents.")
        for document in GROUPED_DOCUMENTS:
            await cognee.add(document)
            print(f"Added document: {document.strip()[:70]}...")

    if enable_steps.get("cognify"):
        await cognee.cognify()
        print("Knowledge graph created with AUTO_LOW_LEVEL_CANONICAL.")

    if enable_steps.get("retriever"):
        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text=(
                "Compare the CV documents and car description documents. Show how they connect "
                "through recurring entity types such as organizations, roles or models, projects "
                "or trims, skills or features, tools or components, metrics, risks, certifications, "
                "budgets, warranties, and timelines."
            ),
            top_k=30,
        )
        pprint(search_results)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)

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
