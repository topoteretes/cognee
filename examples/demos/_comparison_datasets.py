"""Text datasets for the auto-restricted ontology comparison demo.

Each dataset is a list of 5 short text items. Datasets are deliberately
diverse so the approaches can be compared across very different content:
biographical (CVs), procedural (recipes), technical (scientific abstracts),
narrative (historical events).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simple_cognee_example import job_1, job_2, job_3, job_4, job_5


recipe_1 = """
RECIPE: Spaghetti alla Carbonara
Cuisine: Italian (Roman)
Difficulty: Easy
Time: 25 minutes
Serves: 4

Ingredients:
- 400g spaghetti
- 200g guanciale, diced
- 4 large egg yolks
- 100g Pecorino Romano, grated
- Freshly ground black pepper
- Salt for the pasta water

Method:
Bring a large pot of salted water to a boil and cook the spaghetti until al dente.
Fry the guanciale in a skillet over medium heat until crisp, then remove from the heat.
Whisk the egg yolks with the Pecorino Romano and plenty of black pepper.
Drain the pasta, reserving some cooking water.
Combine the pasta with the guanciale off the heat, then add the egg mixture and toss quickly.
The residual heat cooks the eggs into a creamy sauce; add pasta water to loosen.
Serve immediately, topped with extra Pecorino and more black pepper.

Equipment: large pot, skillet, mixing bowl, whisk, colander.
"""

recipe_2 = """
RECIPE: Thai Green Curry with Chicken
Cuisine: Thai
Difficulty: Medium
Time: 40 minutes
Serves: 4

Ingredients:
- 500g chicken thigh, sliced
- 400ml coconut milk
- 3 tbsp green curry paste
- 100g Thai eggplant, quartered
- 1 cup bamboo shoots
- 1 red chili, sliced
- 2 kaffir lime leaves
- 1 tbsp fish sauce
- 1 tsp palm sugar
- Fresh Thai basil
- Jasmine rice for serving

Method:
Heat a wok over medium-high heat and add half the coconut milk, stirring until the oil separates.
Add the green curry paste and fry until fragrant.
Add the chicken and stir until coated and partially cooked through.
Pour in the remaining coconut milk together with the fish sauce, palm sugar, and lime leaves.
Simmer until the chicken is tender, then add the eggplant and bamboo shoots.
Cook until the vegetables soften, then stir in the chili and Thai basil.
Serve over jasmine rice.

Equipment: wok, sharp knife, cutting board, ladle.
"""

recipe_3 = """
RECIPE: Beef Bourguignon
Cuisine: French
Difficulty: Medium
Time: 3 hours
Serves: 6

Ingredients:
- 1.5 kg beef chuck, cubed
- 200g smoked bacon, diced
- 2 onions, chopped
- 3 carrots, sliced
- 4 cloves garlic, minced
- 750ml red Burgundy wine
- 500ml beef stock
- 2 tbsp tomato paste
- 300g pearl onions
- 300g button mushrooms
- Fresh thyme, bay leaves, parsley
- Plain flour, butter, olive oil

Method:
Pat the beef dry and brown the cubes in batches in a Dutch oven.
Render the bacon, then saute the onions, carrots, and garlic until softened.
Return the beef to the pot, dust with flour, and stir for two minutes.
Deglaze with the red wine and add the beef stock, tomato paste, and herbs.
Cover and braise in a 160C oven for two hours.
Saute the pearl onions and mushrooms separately in butter, then fold into the stew.
Continue cooking until the beef is fork-tender.
Serve with mashed potatoes or crusty bread.

Equipment: Dutch oven, skillet, oven, wooden spoon.
"""

recipe_4 = """
RECIPE: Sushi Maki Rolls
Cuisine: Japanese
Difficulty: Medium
Time: 1 hour
Serves: 4

Ingredients:
- 300g sushi rice
- 2 tbsp rice vinegar
- 1 tbsp sugar
- 1 tsp salt
- 4 nori sheets
- 200g sashimi-grade tuna
- 1 cucumber, julienned
- 1 avocado, sliced
- Wasabi paste
- Soy sauce
- Pickled ginger

Method:
Rinse the sushi rice until the water runs clear, then cook in a rice cooker.
While the rice is warm, fold in the vinegar, sugar, and salt mixture.
Lay a sheet of nori on a bamboo mat, shiny side down.
Spread a thin layer of rice across the nori, leaving a gap at the top edge.
Place tuna, cucumber, and avocado in a line across the rice.
Roll tightly with the bamboo mat, sealing the edge with a little water.
Slice the roll into eight pieces with a sharp wet knife.
Serve with wasabi, soy sauce, and pickled ginger.

Equipment: rice cooker, bamboo rolling mat, sharp knife.
"""

recipe_5 = """
RECIPE: Tiramisu
Cuisine: Italian
Difficulty: Easy
Time: 30 minutes plus 4 hours chilling
Serves: 8

Ingredients:
- 6 egg yolks
- 150g caster sugar
- 500g mascarpone cheese
- 300ml double cream
- 400ml strong espresso, cooled
- 4 tbsp Marsala wine or coffee liqueur
- 24 savoiardi (ladyfinger biscuits)
- Cocoa powder for dusting
- Dark chocolate, grated

Method:
Whisk the egg yolks with the sugar until pale and thick.
Fold in the mascarpone until smooth.
Whip the double cream to soft peaks and gently fold it into the mascarpone mixture.
Combine the espresso and Marsala in a shallow dish.
Dip each savoiardi quickly in the coffee mixture and arrange in a serving dish.
Spread half the mascarpone cream over the layer of biscuits.
Repeat with a second layer of dipped biscuits and the remaining cream.
Dust generously with cocoa powder and grated dark chocolate.
Chill for at least four hours before serving.

Equipment: hand mixer, mixing bowls, serving dish, sieve.
"""


science_1 = """
TITLE: AlphaFold - Accurate Protein Structure Prediction with Deep Learning
Authors: John Jumper, Richard Evans, Alexander Pritzel, and colleagues
Institution: DeepMind, London
Year: 2021
Journal: Nature

Abstract:
The protein folding problem has been an outstanding challenge in computational
biology for over fifty years. AlphaFold is a neural network model that predicts
the three-dimensional structure of proteins from their amino acid sequences
with atomic-level accuracy.

The model is trained on a database of known protein structures from the
Protein Data Bank. It uses an attention-based architecture that processes
multiple sequence alignments and pairwise residue features.

In the CASP14 critical assessment, AlphaFold achieved a median accuracy of
0.96 Angstrom on the hardest targets, comparable to experimental methods such
as X-ray crystallography and cryo-electron microscopy. The system has predicted
structures for over 98 percent of human proteins.

This work has accelerated drug discovery, enzyme design, and our understanding
of genetic diseases. The predicted structures are released through the
AlphaFold Protein Structure Database, hosted by EMBL-EBI.

Keywords: deep learning, protein structure, attention mechanism,
computational biology, drug discovery.
"""

science_2 = """
TITLE: Attention Is All You Need - The Transformer Architecture
Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, and colleagues
Institution: Google Brain
Year: 2017
Conference: NeurIPS

Abstract:
Recurrent neural networks have dominated sequence transduction tasks such as
machine translation. We propose the Transformer, a model architecture based
entirely on attention mechanisms, dispensing with recurrence and convolutions.

The model uses self-attention and multi-head attention to compute
representations of its input and output. Position-wise feed-forward networks
process each position independently. Positional encodings inject information
about the order of tokens.

We trained the Transformer on the WMT 2014 English-to-German translation
task. It outperforms the previous best models, achieving a BLEU score of 28.4
with significantly less training time.

The Transformer architecture has since been adopted in BERT, GPT, and modern
large language models. It is now the dominant architecture for natural
language processing and is extending to vision and speech tasks.

Keywords: attention mechanism, sequence modeling, machine translation,
neural networks.
"""

science_3 = """
TITLE: CRISPR-Cas9 - A Programmable Genome Editing System
Authors: Jennifer Doudna, Emmanuelle Charpentier
Institution: UC Berkeley, Max Planck Institute for Infection Biology
Year: 2012
Journal: Science

Abstract:
The CRISPR-Cas9 system is a bacterial immune mechanism that has been adapted
into a versatile tool for genome editing. The system uses a guide RNA to
direct the Cas9 endonuclease to a specific DNA sequence, where it introduces
a double-strand break.

The break is repaired by either non-homologous end joining, which often
produces small insertions or deletions, or homology-directed repair, which
can introduce specific sequence changes when a donor template is provided.

CRISPR-Cas9 has been used to edit genes in bacteria, plants, mice, and human
cells. Clinical trials are underway for sickle cell disease, beta-thalassemia,
and inherited blindness. The technology earned Doudna and Charpentier the
Nobel Prize in Chemistry in 2020.

Ethical concerns remain regarding germline editing, off-target mutations,
and equitable access. The scientific community continues to develop
guidelines for responsible use.

Keywords: CRISPR, Cas9, genome editing, guide RNA, gene therapy.
"""

science_4 = """
TITLE: Higgs Boson Discovery at the Large Hadron Collider
Authors: The ATLAS and CMS collaborations
Institution: CERN, Geneva
Year: 2012
Journal: Physics Letters B

Abstract:
The Higgs boson is a fundamental particle predicted by the Standard Model of
particle physics. It is responsible for giving mass to other elementary
particles through interaction with the Higgs field.

The ATLAS and CMS experiments at the Large Hadron Collider independently
observed a new particle with a mass of approximately 125 GeV. The observation
was based on proton-proton collision data collected at center-of-mass
energies of 7 and 8 TeV.

The new particle decays into pairs of photons, Z bosons, and W bosons in
patterns consistent with the predicted properties of the Higgs boson.
Statistical significance exceeded five standard deviations in both
experiments.

The discovery was announced on 4 July 2012 and earned Peter Higgs and
Francois Englert the Nobel Prize in Physics in 2013. It completed the
Standard Model and opened new investigations into beyond-Standard-Model
physics.

Keywords: particle physics, Higgs boson, Large Hadron Collider, Standard
Model, ATLAS, CMS.
"""

science_5 = """
TITLE: mRNA Vaccines - A New Platform for Pandemic Preparedness
Authors: Katalin Kariko, Drew Weissman
Institution: University of Pennsylvania
Year: 2020
Journal: Nature Reviews Drug Discovery

Abstract:
Messenger RNA vaccines deliver synthetic mRNA encoding a viral antigen into
host cells, which then produce the antigen and trigger an immune response.
The platform offers rapid development and manufacturing compared to
traditional protein-based vaccines.

The mRNA is encapsulated in lipid nanoparticles that protect it from
degradation and facilitate cellular uptake. Modified nucleosides reduce
innate immune activation and improve translation efficiency.

The Pfizer-BioNTech and Moderna mRNA vaccines for SARS-CoV-2 demonstrated
over 94 percent efficacy in preventing symptomatic COVID-19. They were
authorized for emergency use within twelve months of the pandemic
declaration.

Future applications include vaccines against influenza, HIV, and malaria,
together with personalized cancer vaccines. The technology is also being
explored for treating genetic diseases and for protein-replacement
therapies.

Keywords: mRNA vaccine, lipid nanoparticles, SARS-CoV-2, immunology,
mRNA therapeutics.
"""


history_1 = """
EVENT: The Magna Carta
Date: 15 June 1215
Location: Runnymede, England
Key figures: King John of England, Archbishop Stephen Langton, the English barons

Background:
King John faced rebellion from his barons over heavy taxation, military
failures in France, and disputes with the Catholic Church. The Archbishop of
Canterbury, Stephen Langton, mediated between the king and the rebels.

The event:
On 15 June 1215, at the meadow of Runnymede beside the River Thames, King John
affixed his seal to a charter of liberties. The document limited royal
authority, protected church rights, established due process, and restricted
feudal payments to the crown.

Consequences:
The charter was annulled by Pope Innocent III within months, leading to the
First Barons' War. After King John's death in 1216, the charter was reissued
under his son Henry III. It became a foundational document of English
constitutional law and influenced later documents including the United
States Constitution.

Legacy:
The Magna Carta established the principle that the monarch is subject to the
law. It is housed today at the British Library, Salisbury Cathedral, and
Lincoln Castle.
"""

history_2 = """
EVENT: The Fall of Constantinople
Date: 29 May 1453
Location: Constantinople, Byzantine Empire
Key figures: Sultan Mehmed II, Emperor Constantine XI Palaiologos

Background:
The Byzantine Empire had been in decline for centuries, reduced largely to
the city of Constantinople and a few surrounding territories. Sultan Mehmed
II of the Ottoman Empire, only twenty-one years old, sought to capture the
city.

The siege:
Mehmed deployed an army of approximately 80,000 troops and a fleet of
warships. He used massive cannons built by the Hungarian engineer Orban,
which battered the famous Theodosian Walls.

The Byzantine defenders, led by Emperor Constantine XI, numbered around
7,000 men, including Italian mercenaries under Giovanni Giustiniani. After
a fifty-three day siege, the Ottomans breached the walls.

The event:
On 29 May 1453, Ottoman forces entered the city. Emperor Constantine XI died
defending the walls. The Hagia Sophia was converted into a mosque, and
Constantinople became the Ottoman capital, later renamed Istanbul.

Legacy:
The fall ended the Byzantine Empire and marked the beginning of Ottoman
dominance in the eastern Mediterranean. Scholars fleeing the city
contributed to the Italian Renaissance.
"""

history_3 = """
EVENT: The Storming of the Bastille
Date: 14 July 1789
Location: Paris, France
Key figures: Bernard-Rene de Launay, the Marquis de Lafayette, King Louis XVI

Background:
France faced a severe financial crisis and food shortages. King Louis XVI
convened the Estates-General in May 1789. The Third Estate broke away to
form the National Assembly and took the Tennis Court Oath to draft a new
constitution.

The event:
On 14 July 1789, a crowd of Parisian citizens marched to the Bastille, a
medieval fortress and prison that symbolized royal tyranny. The governor,
Bernard-Rene de Launay, commanded a garrison of about 80 men. After
negotiations failed, the crowd stormed the fortress, killing de Launay and
freeing seven prisoners.

Consequences:
The fall of the Bastille triggered the French Revolution. The National
Assembly abolished feudal privileges in August 1789 and adopted the
Declaration of the Rights of Man and of the Citizen. King Louis XVI was
eventually executed by guillotine in January 1793.

Legacy:
14 July is celebrated as Bastille Day, France's national holiday. The event
symbolizes the overthrow of absolutism and the birth of modern republican
ideals.
"""

history_4 = """
EVENT: The Apollo 11 Moon Landing
Date: 20 July 1969
Location: Sea of Tranquility, the Moon
Key figures: Neil Armstrong, Buzz Aldrin, Michael Collins

Background:
The space race between the United States and the Soviet Union began with the
launch of Sputnik in 1957. President John F. Kennedy committed the United
States to landing a man on the Moon before the end of the 1960s in his
speech to Congress in 1961.

The mission:
Apollo 11 launched from Kennedy Space Center on 16 July 1969 atop a Saturn V
rocket. The crew comprised commander Neil Armstrong, lunar module pilot Buzz
Aldrin, and command module pilot Michael Collins. The spacecraft entered
lunar orbit on 19 July.

The event:
On 20 July 1969, the lunar module Eagle separated from the command module
Columbia and descended to the Sea of Tranquility. Neil Armstrong stepped onto
the surface and declared, "That's one small step for man, one giant leap for
mankind." Buzz Aldrin joined him twenty minutes later.

Legacy:
The astronauts collected lunar rock samples, planted the American flag, and
conducted experiments. They returned safely to Earth on 24 July 1969,
splashing down in the Pacific Ocean. The mission ended the space race in
American favor and remains a defining achievement of the twentieth century.
"""

history_5 = """
EVENT: The Fall of the Berlin Wall
Date: 9 November 1989
Location: Berlin, Germany
Key figures: Mikhail Gorbachev, Helmut Kohl, Gunter Schabowski

Background:
The Berlin Wall divided East and West Berlin from 1961, symbolizing the
Cold War divide between communist and capitalist blocs. Soviet leader
Mikhail Gorbachev introduced reforms known as perestroika and glasnost,
which weakened Soviet control over Eastern Europe.

The event:
On the evening of 9 November 1989, East German official Gunter Schabowski
announced new travel regulations at a press conference. When asked when the
rules would take effect, he replied immediately. East Berliners flooded
checkpoints and demanded passage. The border guards, overwhelmed and lacking
clear orders, opened the gates.

Consequences:
Thousands of East and West Berliners climbed onto the wall, celebrating and
chipping away at it with hammers and chisels. The wall was officially
dismantled in the following months. The two German states were reunified on
3 October 1990 under Chancellor Helmut Kohl.

Legacy:
The fall of the Berlin Wall accelerated the collapse of communist regimes
across Eastern Europe. The Soviet Union itself dissolved in 1991. The event
marked the symbolic end of the Cold War.
"""


DATASETS: dict[str, dict] = {
    "cvs": {
        "label": "CVs (job-finder demo)",
        "description": "5 candidate CVs ingested via simple_cognee_example",
        "texts": [job_1, job_2, job_3, job_4, job_5],
    },
    "recipes": {
        "label": "Recipes",
        "description": "5 short recipes across cuisines",
        "texts": [recipe_1, recipe_2, recipe_3, recipe_4, recipe_5],
    },
    "scientific_abstracts": {
        "label": "Scientific abstracts",
        "description": "5 landmark scientific paper summaries",
        "texts": [science_1, science_2, science_3, science_4, science_5],
    },
    "historical_events": {
        "label": "Historical events",
        "description": "5 historical-event narratives (heavy in past-tense verbs)",
        "texts": [history_1, history_2, history_3, history_4, history_5],
    },
}
