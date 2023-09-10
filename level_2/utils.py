import os
from datetime import datetime

from langchain import PromptTemplate, OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import dotenv
dotenv.load_dotenv()

llm_base = OpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4-0613",
        )
def _add_to_episodic(user_input, tasks_list, result_tasks, attention_modulators, params):
    from level_2.level_2_pdf_vectorstore__dlt_contracts import Memory
    memory = Memory(user_id="TestUser")
    class EpisodicTask(BaseModel):
        """Schema for an individual task."""

        task_order: str = Field(
            ..., description="The order at which the task needs to be performed"
        )
        task_name: str = Field(
            None, description="The task that needs to be performed"
        )
        operation: str = Field(None, description="The operation to be performed")
        operation_result: str = Field(
            None, description="The result of the operation"
        )

    class EpisodicList(BaseModel):
        """Schema for the record containing a list of tasks."""

        tasks: List[EpisodicTask] = Field(..., description="List of tasks")
        start_date: str = Field(
            ..., description="The order at which the task needs to be performed"
        )
        end_date: str = Field(
            ..., description="The order at which the task needs to be performed"
        )
        user_query: str = Field(
            ..., description="The order at which the task needs to be performed"
        )
        attention_modulators: str = Field(..., description="List of attention modulators")

    parser = PydanticOutputParser(pydantic_object=EpisodicList)
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    prompt = PromptTemplate(
        template="Format the result.\n{format_instructions}\nOriginal query is: {query}\n Steps are: {steps}, buffer is: {buffer}, date is:{date}, attention modulators are: {attention_modulators} \n",
        input_variables=["query", "steps", "buffer", "date", "attention_modulators"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(
        query=user_input, steps=str(tasks_list)
        , buffer=str(result_tasks), date=date, attention_modulators=attention_modulators
    )

    # return "a few things to do like load episodic memory in a structured format"
    output = llm_base(_input.to_string())
    result_parsing = parser.parse(output)
    lookup_value = await memory.add_memories(
        observation=str(result_parsing.json()), params=params, namespace='EPISODICMEMORY'
    )


def add_to_buffer():
    pass


def delete_from_buffer():
    from level_2.level_2_pdf_vectorstore__dlt_contracts import Memory
    memory = Memory(user_id="TestUser")
    memory._delete_buffer_memory()

def delete_from_episodic():
    from level_2.level_2_pdf_vectorstore__dlt_contracts import Memory
    memory = Memory(user_id="TestUser")
    memory._delete_episodic_memory()

if __name__ == "__main__":

    params = {
        "version": "1.0",
        "agreement_id": "AG123456",
        "privacy_policy": "https://example.com/privacy",
        "terms_of_service": "https://example.com/terms",
        "format": "json",
        "schema_version": "1.1",
        "checksum": "a1b2c3d4e5f6",
        "owner": "John Doe",
        "license": "MIT",
        "validity_start": "2023-08-01",
        "validity_end": "2024-07-31",
    }
    loader_settings =  {
    "format": "PDF",
    "source": "url",
    "path": "https://www.ibiblio.org/ebooks/London/Call%20of%20Wild.pdf"
    }
    modulator = {"relevance": 0.0, "saliency": 0.0, "frequency": 0.0}
    user_input = "I want to know how does Buck adapt to life in the wild"
    # tasks_list = """tasks": [{"task_order": "1", "task_name": "Fetch Information", "operation": "fetch from vector store", "original_query": "I want to know how does Buck adapt to life in the wild"]"""
    out_tasks = """here are the result_tasks [{'task_order': '1', 'task_name': 'Fetch Information', 'operation': 'fetch from vector store', 'original_query': 'I want to know how does Buck adapt to life in the wild'}, {'docs': [{'semantic_search_term': "Buck's adaptation to wild life", 'document_content': 'THE  CALL  OF  THE  WILD 30 \nout of his desire for mastery. He was preëminently cunning, and could \nbide his time with a patience that was nothing less than primitive.  \nIt was inevitable that the clash for leadership should come. Buck \nwanted it. He wanted it because it was his nature, because he had been \ngripped tight by that nameless, incomprehensible pride of the trail and \ntrace—that pride which holds dogs in the toil to the last gasp, which \nlures them to die joyfully in the harness, and breaks their hearts if they \nare cut out of the harness. This was the pride of Dave as wheel-dog, of \nSol-leks as he pulled with all his strength; the pride that laid hold of \nthem at break of camp, transforming them from sour and sullen brutes \ninto straining, eager, ambitious creatures; the pride that spurred them on \nall day and dropped them at pitch of camp at night, letting them fall back \ninto gloomy unrest and uncontent. This was the pride that bore up Spitz \nand made him thrash the sled-dogs who blundered and shirked in the \ntraces or hid away at harness-up time in the morning. Likewise it was \nthis pride that made him fear Buck as a possible lead-dog. And this was \nBuck’s pride, too.  \nHe openly threatened the other’s leadership. He came between him \nand the shirks he should have punished. And he did it deliberately. One \nnight there was a heavy snowfall, and in the morning Pike, the \nmalingerer, did not appear. He was securely hidden in his nest under a \nfoot of snow. François called him and sought him in vain. Spitz was wild \nwith wrath. He raged through the camp, smelling and digging in every \nlikely place, snarling so frightfully that Pike heard and shivered in his \nhiding-place.  \nBut when he was at last unearthed, and Spitz flew at him to punish \nhim, Buck flew, with equal rage, in between. So unexpected was it, and \nso shrewdly managed, that Spitz was hurled backward and off his feet. \nPike, who had been trembling abjectly, took heart at this open mutiny, \nand sprang upon his overthrown leader. Buck, to whom fairplay was a \nforgotten code, likewise sprang upon Spitz. But François, chuckling at \nthe incident while unswerving in the administration of justice, brought \nhis lash down upon Buck with all his might. This failed to drive Buck \nfrom his prostrate rival, and the butt of the whip was brought into play. \nHalf-stunned by the blow, Buck was knocked backward and the lash laid', 'document_relevance': '0.75', 'attention_modulators_list': [{'frequency': 'High', 'saliency': 'High', 'relevance': 'High'}]}], 'user_query': 'I want to know how does Buck adapt to life in the wild and then have that info translated to german'}, {'task_order': '2', 'task_name': 'Translate Information', 'operation': 'translate', 'original_query': 'then have that info translated to german'}, 'DER RUF DER WILDNIS 30\naus seinem Wunsch nach Meisterschaft. Er war überaus schlau und konnte es\nwartete seine Zeit mit einer Geduld ab, die geradezu primitiv war.\nEs war unvermeidlich, dass es zu einem Kampf um die Führung kam. Bock\nwollte es. Er wollte es, weil es in seiner Natur lag, weil er es gewesen war\nfestgehalten von diesem namenlosen, unverständlichen Stolz des Weges und\nSpur – dieser Stolz, der Hunde bis zum letzten Atemzug in der Mühsal hält, der\nlockt sie dazu, freudig im Geschirr zu sterben, und bricht ihnen das Herz, wenn sie es tun\nwerden aus dem Kabelbaum herausgeschnitten. Das war der Stolz von Dave als Radhund\nSol-leks, als er mit aller Kraft zog; der Stolz, der mich ergriff\nsie beim Abbruch des Lagers und verwandelte sie in mürrische und mürrische Bestien\nin anstrengende, eifrige, ehrgeizige Wesen; der Stolz, der sie anspornte\nden ganzen Tag und setzte sie nachts auf dem Stellplatz des Lagers ab und ließ sie zurückfallen\nin düstere Unruhe und Unzufriedenheit. Das war der Stolz, der Spitz trug\nund ließ ihn die Schlittenhunde verprügeln, die in der Gegend herumstolperten und sich scheuten\nSpuren hinterlassen oder sich morgens beim Angurten versteckt haben. Ebenso war es\nDieser Stolz ließ ihn Buck als möglichen Leithund fürchten. Und das war\nAuch Bucks Stolz.\nEr bedrohte offen die Führung des anderen. Er kam zwischen ihn\nund die Schirks hätte er bestrafen sollen. Und er hat es absichtlich getan. Eins\nNachts gab es starken Schneefall und am Morgen Pike, der\nSimulant, erschien nicht. Er war sicher in seinem Nest unter einer Decke versteckt\nFuß Schnee. François rief ihn an und suchte ihn vergeblich. Spitz war wild\nmit Zorn. Er tobte durch das Lager, schnupperte und wühlte darin herum\nWahrscheinlicher Ort, er knurrte so schrecklich, dass Pike es hörte und in seinem Kopf zitterte\nVersteck.\nAber als er endlich ausgegraben wurde, flog Spitz auf ihn zu, um ihn zu bestrafen\nihm, Buck flog mit der gleichen Wut dazwischen. So unerwartet war es, und\nEs gelang ihm so geschickt, dass Spitz nach hinten geschleudert wurde und von den Füßen fiel.\nPike, der erbärmlich gezittert hatte, fasste angesichts dieser offenen Meuterei Mut.\nund sprang auf seinen gestürzten Anführer. Buck, für den Fairplay wichtig war\nvergessener Code, sprang ebenfalls Spitz auf. Aber François kicherte\nder Vorfall, während unerschütterlich in der Rechtspflege, gebracht\nEr schlug mit aller Kraft auf Buck ein. Das gelang Buck nicht\nvon seinem am Boden liegenden Rivalen, und der Peitschenkolben wurde ins Spiel gebracht.\nVon dem Schlag halb betäubt, wurde Buck nach hinten geschleudert und mit der Peitsche niedergeschlagen']"""

    # _add_to_episodic(user_input=user_input, result_tasks=out_tasks, modulator=None, params=params)


