from cognee.eval_framework.benchmark_adapters.hotpot_qa_adapter import HotpotQAAdapter
import json

INSTANCE_FILTER = [
    "5a8e341c5542995085b373d6",
    "5ab2a308554299340b52553b",
    "5a79c1095542996c55b2dc62",
    "5a8c52685542995e66a475bb",
    "5a734dad5542994cef4bc522",
    "5a74ab6055429916b01641b9",
    "5ae2661d554299495565da60",
    "5a88dcf9554299206df2b383",
    "5ab8179f5542990e739ec817",
    "5a812d0555429938b61422e1",
    "5a79be0e5542994f819ef084",
    "5a875b755542996e4f308796",
    "5ae675245542991bbc9760dc",
    "5ab819065542995dae37ea3c",
    "5a74d64055429916b0164223",
    "5abfea825542994516f45527",
    "5ac279345542990b17b153b0",
    "5ab3c48755429969a97a81b8",
    "5adf35935542993344016c36",
    "5a83d0845542996488c2e4e6",
    "5a7af32e55429931da12c99c",
    "5a7c9ead5542990527d554e4",
    "5ae12aa6554299422ee99617",
    "5a710a915542994082a3e504",
]


def main():
    hotpot_qa_adapter = HotpotQAAdapter()
    corpus_list, qa_pairs = hotpot_qa_adapter.load_corpus(instance_filter=INSTANCE_FILTER)
    json.dump(corpus_list, open("hotpot_qa_24_corpus.json", "w"), indent=2)
    json.dump(qa_pairs, open("hotpot_qa_24_qa_pairs.json", "w"), indent=2)


if __name__ == "__main__":
    main()
