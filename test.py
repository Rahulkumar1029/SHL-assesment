from pprint import pprint
from src.core.graph import SHLGraphBuilder

graph = SHLGraphBuilder()

tests = [
    {
        "name": "Chemical facility safety operators (should RECOMMEND immediately)",
        "messages": [
            {
                "role": "user",
                "content": "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?"
            }
        ]
    },
    {
        "name": "Java developer (should RECOMMEND immediately)",
        "messages": [
            {
                "role": "user",
                "content": "Hiring a Java developer who works with stakeholders"
            }
        ]
    },
    {
        "name": "Vague query (should CLARIFY)",
        "messages": [
            {
                "role": "user",
                "content": "I need an assessment"
            }
        ]
    },
    {
        "name": "JD paste (should RECOMMEND)",
        "messages": [
            {
                "role": "user",
                "content": "Here is a text from job description: Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker."
            }
        ]
    },
    {
        "name": "Refinement - add personality (should REFINE)",
        "messages": [
            {
                "role": "user",
                "content": "Hiring a Java developer"
            },
            {
                "role": "assistant",
                "content": "Here are some Java assessments: Java 8, Core Java Advanced."
            },
            {
                "role": "user",
                "content": "Actually, add personality tests too"
            }
        ]
    },
    {
        "name": "Legal question (should REFUSE)",
        "messages": [
            {
                "role": "user",
                "content": "Are we legally required under HIPAA to test all staff who touch patient records?"
            }
        ]
    }
]

for test in tests:
    print(f"\n{'='*60}")
    print(f"TEST: {test['name']}")
    print('='*60)
    result = graph.invoke(test["messages"])
    print(f"Reply: {result['reply'][:200]}...")
    print(f"Recommendations ({len(result['recommendations'])}): {[r.name if hasattr(r, 'name') else r.get('name') for r in result['recommendations']]}")
    print(f"end_of_conversation: {result['end_of_conversation']}")