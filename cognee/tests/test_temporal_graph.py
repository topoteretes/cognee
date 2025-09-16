import asyncio
import cognee
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever

from cognee.shared.logging_utils import setup_logging, INFO
from cognee.tasks.temporal_graph.models import Timestamp
from cognee.api.v1.search import SearchType
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from collections import Counter
from cognee.modules.engine.utils.generate_timestamp_datapoint import date_to_int

logger = get_logger()

biography_1 = """
    Attaphol Buspakom Attaphol Buspakom ( ; ) , nicknamed Tak ( ; ) ; 1 October 1962 – 16 April 2015 ) was a Thai national and football coach . He was given the role at Muangthong United and Buriram United after TTM Samut Sakhon folded after the 2009 season . He played for the Thailand national football team , appearing in several FIFA World Cup qualifying matches .

    Club career .
    Attaphol began his career as a player at Thai Port FC Authority of Thailand in 1985 . In his first year , he won his first championship with the club . He played for the club until 1989 and in 1987 also won the Queens Cup . He then moved to Malaysia for two seasons for Pahang FA , then return to Thailand to his former club . His time from 1991 to 1994 was marked by less success than in his first stay at Port Authority . From 1994 to 1996 he played for Pahang again and this time he was able to win with the club , the Malaysia Super League and also reached the final of the Malaysia Cup and the Malaysia FA Cup . Both cup finals but lost . Back in Thailand , he let end his playing career at FC Stock Exchange of Thailand , with which he once again runner‑up in 1996-97 . In 1998 , he finished his career .

    International career .
    For the Thailand national football team Attaphol played between 1985 and 1998 a total of 85 games and scored 13 results . In 1992 , he participated with the team in the finals of the Asian Cup . He also stood in various cadres to qualifications to FIFA World Cup .

    Coaching career .
    Bec Tero Sasana .
    In BEC Tero Sasana F.C . began his coaching career in 2001 for him , first as assistant coach . He took over the reigning champions of the Thai League T1 , after his predecessor Pichai Pituwong resigned from his post . It was his first coach station and he had the difficult task of leading the club through the new AFC Champions League . He could accomplish this task with flying colors and even led the club to the finals . The finale , then still played in home and away matches , was lost with 1:2 at the end against Al Ain FC . Attaphol is and was next to Charnwit Polcheewin the only coach who managed a club from Thailand to lead to the final of the AFC Champions League . 2002-03 and 2003-04 he won with the club also two runner‑up . In his team , which reached the final of the Champions League , were a number of exceptional players like Therdsak Chaiman , Worrawoot Srimaka , Dusit Chalermsan and Anurak Srikerd .

    Geylang United / Krung Thai Bank .
    In 2006 , he went to Singapore in the S‑League to Geylang United He was released after a few months due to lack of success . In 2008 , he took over as coach at Krung Thai Bank F.C. , where he had almost a similar task , as a few years earlier by BEC‑Tero . As vice‑champion of the club was also qualified for the AFC Champions League . However , he failed to lead the team through the group stage of the season 2008 and beyond . With the Kashima Antlers of Japan and Beijing Guoan F.C . athletic competition was too great . One of the highlights was put under his leadership , yet the club . In the group match against the Vietnam club Nam Dinh F.C . his team won with 9-1 , but also lost four weeks later with 1-8 against Kashima Antlers . At the end of the National Football League season , he reached the Krung Thai 6th Table space . The Erstligalizenz the club was sold at the end of the season at the Bangkok Glass F.C. . Attaphol finished his coaching career with the club and accepted an offer of TTM Samutsakorn . After only a short time in office

    Muangthong United .
    In 2009 , he received an offer from Muangthong United F.C. , which he accepted and changed . He can champion Muang Thong United for 2009 Thai Premier League and Attaphol won Coach of The year for Thai Premier League and he was able to lead Muang Thong United to play AFC Champions League qualifying play‑off for the first in the clubs history .

    Buriram United .
    In 2010 Buspakom moved from Muangthong United to Buriram United F.C. . He received Coach of the Month in Thai Premier League 2 time in June and October . In 2011 , he led Buriram United win 2011 Thai Premier League second time for club and set a record with the most points in the Thai League T1 for 85 point and He led Buriram win 2011 Thai FA Cup by beat Muangthong United F.C . 1‑0 and he led Buriram win 2011 Thai League Cup by beat Thai Port F.C . 2‑0 . In 2012 , he led Buriram United to the 2012 AFC Champions League group stage . Buriram along with Guangzhou Evergrande F.C . from China , Kashiwa Reysol from Japan and Jeonbuk Hyundai Motors which are all champions from their country . In the first match of Buriram they beat Kashiwa 3‑2 and Second Match they beat Guangzhou 1‑2 at the Tianhe Stadium . Before losing to Jeonbuk 0‑2 and 3‑2 with lose Kashiwa and Guangzhou 1‑0 and 1‑2 respectively and Thai Premier League Attaphol lead Buriram end 4th for table with win 2012 Thai FA Cup and 2012 Thai League Cup .

    Bangkok Glass .
    In 2013 , he moved from Buriram United to Bangkok Glass F.C. .

    Individual
    - Thai Premier League Coach of the Year ( 3 ) : 2001-02 , 2009 , 2013
    """


biography_2 = """
    Arnulf Øverland Ole Peter Arnulf Øverland ( 27 April 1889 – 25 March 1968 ) was a Norwegian poet and artist . He is principally known for his poetry which served to inspire the Norwegian resistance movement during the German occupation of Norway during World War II .

    Biography .
    Øverland was born in Kristiansund and raised in Bergen . His parents were Peter Anton Øverland ( 1852–1906 ) and Hanna Hage ( 1854–1939 ) . The early death of his father , left the family economically stressed . He was able to attend Bergen Cathedral School and in 1904 Kristiania Cathedral School . He graduated in 1907 and for a time studied philology at University of Kristiania . Øverland published his first collection of poems ( 1911 ) .

    Øverland became a communist sympathizer from the early 1920s and became a member of Mot Dag . He also served as chairman of the Norwegian Students Society 1923–28 . He changed his stand in 1937 , partly as an expression of dissent against the ongoing Moscow Trials . He was an avid opponent of Nazism and in 1936 he wrote the poem Du må ikke sove which was printed in the journal Samtiden . It ends with . ( I thought: : Something is imminent . Our era is over – Europe’s on fire! ) . Probably the most famous line of the poem is ( You mustnt endure so well the injustice that doesnt affect you yourself! )

    During the German occupation of Norway from 1940 in World War II , he wrote to inspire the Norwegian resistance movement . He wrote a series of poems which were clandestinely distributed , leading to the arrest of both him and his future wife Margrete Aamot Øverland in 1941 . Arnulf Øverland was held first in the prison camp of Grini before being transferred to Sachsenhausen concentration camp in Germany . He spent a four‑year imprisonment until the liberation of Norway in 1945 . His poems were later collected in Vi overlever alt and published in 1945 .

    Øverland played an important role in the Norwegian language struggle in the post‑war era . He became a noted supporter for the conservative written form of Norwegian called Riksmål , he was president of Riksmålsforbundet ( an organization in support of Riksmål ) from 1947 to 1956 . In addition , Øverland adhered to the traditionalist style of writing , criticising modernist poetry on several occasions . His speech Tungetale fra parnasset , published in Arbeiderbladet in 1954 , initiated the so‑called Glossolalia debate .

    Personal life .
    In 1918 he had married the singer Hildur Arntzen ( 1888–1957 ) . Their marriage was dissolved in 1939 . In 1940 , he married Bartholine Eufemia Leganger ( 1903–1995 ) . They separated shortly after , and were officially divorced in 1945 . Øverland was married to journalist Margrete Aamot Øverland ( 1913–1978 ) during June 1945 . In 1946 , the Norwegian Parliament arranged for Arnulf and Margrete Aamot Øverland to reside at the Grotten . He lived there until his death in 1968 and she lived there for another ten years until her death in 1978 . Arnulf Øverland was buried at Vår Frelsers Gravlund in Oslo . Joseph Grimeland designed the bust of Arnulf Øverland ( bronze , 1970 ) at his grave site .

    Selected Works .
    - Den ensomme fest ( 1911 )
    - Berget det blå ( 1927 )
    - En Hustavle ( 1929 )
    - Den røde front ( 1937 )
    - Vi overlever alt ( 1945 )
    - Sverdet bak døren ( 1956 )
    - Livets minutter ( 1965 )

    Awards .
    - Gyldendals Endowment ( 1935 )
    - Dobloug Prize ( 1951 )
    - Mads Wiel Nygaards legat ( 1961 )
    """


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add([biography_1, biography_2])

    await cognee.cognify(temporal_cognify=True)

    graph_engine = await get_graph_engine()
    graph = await graph_engine.get_graph_data()

    type_counts = Counter(node_data[1].get("type", {}) for node_data in graph[0])

    edge_type_counts = Counter(edge_type[2] for edge_type in graph[1])

    # Graph structure test
    assert type_counts.get("TextDocument", 0) == 2, (
        f"Expected exactly one TextDocument, but found {type_counts.get('TextDocument', 0)}"
    )

    assert type_counts.get("DocumentChunk", 0) == 2, (
        f"Expected exactly one DocumentChunk, but found {type_counts.get('DocumentChunk', 0)}"
    )

    assert type_counts.get("Entity", 0) >= 10, (
        f"Expected multiple entities (assert is set to 20), but found {type_counts.get('Entity', 0)}"
    )

    assert type_counts.get("EntityType", 0) >= 2, (
        f"Expected multiple entity types, but found {type_counts.get('EntityType', 0)}"
    )

    assert type_counts.get("Event", 0) >= 10, (
        f"Expected multiple events (assert is set to 20), but found {type_counts.get('Event', 0)}"
    )

    assert type_counts.get("Timestamp", 0) >= 10, (
        f"Expected multiple timestamps (assert is set to 10), but found {type_counts.get('Timestamp', 0)}"
    )

    assert edge_type_counts.get("contains", 0) >= 10, (
        f"Expected multiple 'contains' edge, but found {edge_type_counts.get('contains', 0)}"
    )

    assert edge_type_counts.get("is_a", 0) >= 10, (
        f"Expected multiple 'is_a' edge, but found {edge_type_counts.get('is_a', 0)}"
    )

    retriever = TemporalRetriever()

    result_between = await retriever.extract_time_from_query("What happened between 1890 and 1900?")

    assert result_between[1]
    assert result_between[0]


if __name__ == "__main__":
    logger = setup_logging(log_level=INFO)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
