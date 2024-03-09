import asyncio
import logging

from langchain.prompts import ChatPromptTemplate
import json
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
import re

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
import instructor
from openai import OpenAI


aclient = instructor.patch(OpenAI())

from typing import Optional, List, Type
from pydantic import BaseModel, Field

from cognitive_architecture.modules.cognify.llm.classify_content import classify_into_categories
from cognitive_architecture.modules.cognify.llm.content_to_cog_layers import content_to_cog_layers
from cognitive_architecture.modules.cognify.llm.content_to_propositions import generate_graph
from cognitive_architecture.shared.data_models import DefaultContentPrediction,  KnowledgeGraph, DefaultCognitiveLayer



async def cognify():
    """This function is responsible for the cognitive processing of the content."""
    # Load the content from the text file


    # Classify the content into categories
    input_article_one = """ In the nicest possible way, Britons have always been a bit silly about animals. “Keeping pets, for the English, is not so much a leisure activity as it is an entire way of life,” wrote the anthropologist Kate Fox in Watching the English, nearly 20 years ago. Our dogs, in particular, have been an acceptable outlet for emotions and impulses we otherwise keep strictly controlled – our latent desire to be demonstratively affectionate, to be silly and chat to strangers. If this seems like an exaggeration, consider the different reactions you’d get if you struck up a conversation with someone in a park with a dog, versus someone on the train.

    Indeed, British society has been set up to accommodate these four-legged ambassadors. In the UK – unlike Australia, say, or New Zealand – dogs are not just permitted on public transport but often openly encouraged. Many pubs and shops display waggish signs, reading, “Dogs welcome, people tolerated”, and have treat jars on their counters. The other day, as I was waiting outside a cafe with a friend’s dog, the barista urged me to bring her inside.

    For years, Britons’ non-partisan passion for animals has been consistent amid dwindling common ground. But lately, rather than bringing out the best in us, our relationship with dogs is increasingly revealing us at our worst – and our supposed “best friends” are paying the price.

    As with so many latent traits in the national psyche, it all came unleashed with the pandemic, when many people thought they might as well make the most of all that time at home and in local parks with a dog. Between 2019 and 2022, the number of pet dogs in the UK rose from about nine million to 13 million. But there’s long been a seasonal surge around this time of year, substantial enough for the Dogs Trust charity to coin its famous slogan back in 1978: “A dog is for life, not just for Christmas.”

    Green spaces, meanwhile, have been steadily declining, and now many of us have returned to the office, just as those “pandemic dogs” are entering their troublesome teens. It’s a combustible combination and we are already seeing the results: the number of dog attacks recorded by police in England and Wales rose by more than a third between 2018 and 2022.

    At the same time, sites such as Pets4Homes.co.uk are replete with listings for dogs that, their owners accept “with deep regret”, are no longer suited to their lifestyles now that lockdown is over. It may have felt as if it would go on for ever, but was there ever any suggestion it was going to last the average dog’s lifespan of a decade?

    Living beings are being downgraded to mere commodities. You can see it reflected the “designer” breeds currently in fashion, the French bulldogs and pugs that look cute but spend their entire lives in discomfort. American XL bully dogs, now so controversial, are often sought after as a signifier of masculinity: roping an entire other life in service of our egos. Historically, many of Britain’s most popular breeds evolved to hunt vermin, retrieve game, herd, or otherwise do a specific job alongside humans; these days we are breeding and buying them for their aesthetic appeal.

    Underpinning this is a shift to what was long disdained as the “American” approach: treating pets as substitutes for children. In the past in Britain, dogs were treasured on their own terms, for the qualities that made them dogs, and as such, sometimes better than people: their friendliness and trustingness and how they opened up the world for us. They were indulged, certainly – by allowing them on to the sofa or in our beds, for instance, when we’d sworn we never would – but in ways that did not negate or deny their essential otherness.

    Now we have more dogs of such ludicrous proportions, they struggle to function as dogs at all – and we treat them accordingly, indulging them as we would ourselves: by buying unnecessary things. The total spend on pets in the UK has more than doubled in the past decade, reaching nearly £10bn last year. That huge rise has not just come from essentials: figures from the marketing agency Mintel suggest that one in five UK owners like their pet to “keep up with the latest trends” in grooming or, heaven forbid, outfits.

    These days pet “boutiques” – like the one that recently opened on my street in Norwich, selling “cold-pressed” dog treats, “paw and nose balms” and spa services – are a widespread sign of gentrification. But it’s not just wealthier areas: this summer in Great Yarmouth, one of the most deprived towns in the country, I noticed seaside stalls selling not one but two brands of ice-cream for dogs.

    It suggests dog-lovers have become untethered from their companions’ desires, let alone their needs. Let’s be honest: most dogs would be thrilled to bits to be eating a paper bag, or even their own faeces. And although they are certainly delighted by ice-cream, they don’t need it. But the ways we ourselves find solace – in consumption, by indulging our simian “treat brain” with things that we don’t need and/or aren’t good for us – we have simply extended to our pets.

    It’s hard not to see the rise in dog-friendly restaurants, cinema screenings and even churches as similar to the ludicrous expenditure: a way to placate the two-legged being on the end of the lead (regardless of the experience of others in the vicinity).

    Meanwhile, many dogs suffer daily deprivation, their worlds made small and monotonous by our busy modern schedules. These are social animals: it’s not natural for them to live without other dogs, let alone in an empty house for eight hours a day, Monday to Friday. If we are besieged by badly behaved dogs, the cause isn’t hard to pinpoint. Many behavioural problems can be alleviated and even addressed by sufficient exercise, supervision and consistent routines, but instead of organising our lives so that our pets may thrive, we show our love with a Halloween-themed cookie, or a new outfit for Instagram likes.

    It’s easy to forget that we are sharing our homes with a descendant of the wolf when it is dressed in sheep’s clothing; but the more we learn about animals, the clearer it becomes that our treatment of them, simultaneously adoring and alienated, means they are leading strange, unsatisfying simulacra of the lives they ought to lead.

    But for as long as we choose to share our lives with pets, the bar should be the same as for any relationship we value: being prepared to make sacrifices for their wellbeing, prioritising quality time and care, and loving them as they are – not for how they reflect on us, or how we’d like them to be.


    """
    required_layers_one = await classify_into_categories(input_article_one, "classify_content.txt",
                                                         DefaultContentPrediction)

    def transform_dict(original):
        # Extract the first subclass from the list (assuming there could be more)
        subclass_enum = original['label']['subclass'][0]

        # The data type is derived from 'type' and converted to lowercase
        data_type = original['label']['type'].lower()

        # The context name is the name of the Enum member (e.g., 'NEWS_STORIES')
        context_name = subclass_enum.name.replace('_', ' ').title()

        # The layer name is the value of the Enum member (e.g., 'News stories and blog posts')
        layer_name = subclass_enum.value

        # Construct the new dictionary
        new_dict = {
            'data_type': data_type,
            'context_name': data_type.upper(),  # llm context classification
            'layer_name': layer_name  # llm layer classification
        }

        return new_dict

    # Transform the original dictionary
    transformed_dict_1 = transform_dict(required_layers_one.dict())
    cognitive_layers_one = await content_to_cog_layers("generate_cog_layers.txt", transformed_dict_1,
                                                       response_model=DefaultCognitiveLayer)
    cognitive_layers_one = [layer_subgroup.name for layer_subgroup in cognitive_layers_one.cognitive_layers]

    async def generate_graphs_for_all_layers(text_input: str, layers: List[str], response_model: Type[BaseModel]):
        tasks = [generate_graph(text_input, "generate_graph_prompt.txt", {'layer': layer}, response_model) for layer in
                 layers]
        return await asyncio.gather(*tasks)

    # Execute the async function and print results for each set of layers
    async def async_graph_per_layer(text_input: str, cognitive_layers: List[str]):
        graphs = await generate_graphs_for_all_layers(text_input, cognitive_layers, KnowledgeGraph)
        # for layer, graph in zip(cognitive_layers, graphs):
        #     print(f"{layer}: {graph}")
        return graphs

    # Run the async function for each set of cognitive layers
    layer_1_graph = await async_graph_per_layer(input_article_one, cognitive_layers_one)