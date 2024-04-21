import dspy
from cognee.modules.data.extraction.knowledge_graph.extract_knowledge_graph_module import ExtractKnowledgeGraph
from cognee.root_dir import get_absolute_path
from cognee.config import Config

config = Config()
config.load()

def run():
    gpt4 = dspy.OpenAI(model = config.openai_model, api_key = config.openai_key, model_type = "chat", max_tokens = 4096)
    compiled_extract_knowledge_graph = ExtractKnowledgeGraph(lm = gpt4)
    compiled_extract_knowledge_graph.load(get_absolute_path("./programs/extract_knowledge_graph/extract_knowledge_graph.json"))

    text = """The 1985 FA Charity Shield (also known as the General Motors FA
              Charity Shield for sponsorship reasons) was the 63rd FA Charity Shield,
              an annual football match played between the winners of the previous
              season's First Division and FA Cup competitions. The match was played on
              10 August 1985 at Wembley Stadium and contested by Everton,
              who had won the 1984\u201385 First Division, and Manchester United,
              who had won the 1984\u201385 FA Cup. Everton won 2\u20130 with goals from
              Trevor Steven and Adrian Heath. Trevor Steven put Everton into the lead
              when he swept home from six yards after a cross from the left in the first half.
              The second goal came in the second half when Manchester United goalkeeper
              Gary Bailey dropped a cross from the left to allow Adrian Heath to tip the
              ball past him into the left corner of the net.\r\nThe 1995 FA Charity Shield
              (also known as the Littlewoods FA Charity Shield for sponsorship reasons) was the
              73rd FA Charity Shield, an annual football match played between the winners of
              the previous season's Premier League and FA Cup competitions. The match was
              played on 13 August 1995 at Wembley Stadium and contested by Blackburn Rovers,
              who had won the Premier League and FA Cup winners Everton. It was Blackburn's
              second successive Charity Shield appearance, while Everton were appearing in
              their eleventh and their first since 1987. Everton won the match 1\u20130
              with a goal from Vinny Samways when he caught Tim Flowers off his line and
              lifted the ball over him from the left of the penalty area and into the right
              corner of the net. Dave Watson lifted the trophy for Everton.\r\nThe 1972 FA
              Charity Shield was contested between Manchester City and Aston Villa.\r\nThe
              1997 FA Charity Shield (known as the Littlewoods FA Charity Shield for
              sponsorship reasons) was the 75th FA Charity Shield, an annual football match
              played between the winners of the previous season's Premier League and
              FA Cup competitions. The match was played on 3 August 1997 at Wembley Stadium
              and contested by Manchester United, who had won the 1996\u201397 FA Premier League,
              and Chelsea, who had won the 1996\u201397 FA Cup. Manchester United won the match
              4\u20132 on penalties after the match had finished at 1\u20131 after 90 minutes.
              \r\nThe 1956 FA Charity Shield was the 34th FA Charity Shield, an annual football
              match held between the winners of the previous season's Football League and
              FA Cup competitions. The match was contested by Manchester United, who had won
              the 1955\u201356 Football League, and Manchester City, who had won the
              1955\u201356 FA Cup, at Maine Road, Manchester, on 24 October 1956. Manchester
              United won the match 1\u20130, Dennis Viollet scoring the winning goal.
              Manchester United goalkeeper David Gaskell made his debut for the club during
              the game, taking the place of injured goalkeeper Ray Wood, and, at the age of
              16 years and 19 days, became the youngest player ever to play for the club.
              \r\nThe 1937 FA Charity Shield was the 24th FA Charity Shield, a football match
              between the winners of the previous season's First Division and FA Cup competitions.
              The match was contested by league champions Manchester City and FA Cup winners
              Sunderland, and was played at Maine Road, the home ground of Manchester City.
              Manchester City won the game, 2\u20130.\r\nThe 2000 FA Charity Shield (also known
              as the One 2 One FA Charity Shield for sponsorship reasons) was the
              78th FA Charity Shield, an annual football match played between the winners
              of the previous season's Premier League and FA Cup competitions. The match
              was played between Manchester United, who won the 1999\u20132000 Premier League,
              and Chelsea, who won the 1999\u20132000 FA Cup, and resulted in a 2\u20130 Chelsea win.
              The goals were scored by Jimmy Floyd Hasselbaink and Mario Melchiot. Roy Keane
              was sent off for a challenge on Gustavo Poyet and was the last person to be
              sent off at the old Wembley Stadium.\r\nThe 2001 FA Charity Shield (also known
              as the One 2 One FA Charity Shield for sponsorship reasons) was the 79th FA Charity Shield,
              an annual football match played between the winners of the previous season's
              Premier League and FA Cup. The match was contested between Liverpool, winners of
              the 2000\u201301 FA Cup and Manchester United, who won the 2000\u201301 Premier
              League on 12 August 2001. It was the first Shield match to be held at the
              Millennium Stadium following the closure of Wembley Stadium for reconstruction.
              \r\nAston Villa Football Club ( ; nicknamed Villa, The Villa, The Villans
              and The Lions) is a professional football club in Aston, Birmingham, that plays
              in the Championship, the second level of English football. Founded in 1874,
              they have played at their current home ground, Villa Park, since 1897. Aston Villa
              were one of the founder members of the Football League in 1888 and of the
              Premier League in 1992.\r\nThe 1996 FA Charity Shield (also known as the
              Littlewoods FA Charity Shield for sponsorship reasons) was the 74th FA Charity Shield,
              an annual football match played between the winners of the previous season's Premier
              League and FA Cup competitions. The match was played on 11 August 1996 at Wembley
              Stadium and contested by Manchester United, who had won the Double of Premier League
              and FA Cup in 1995\u201396, and Newcastle United, who had finished as runners-up
              in the Premier League. Manchester United won the match 4\u20130 with goals from
              Eric Cantona, Nicky Butt, David Beckham and Roy Keane."""

    prediction = compiled_extract_knowledge_graph(context = text, question = "")

    print(prediction.graph)

if __name__ == "__main__":
    run()
