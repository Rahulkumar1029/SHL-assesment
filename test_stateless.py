import json
from src.core.graph import SHLGraphBuilder

def run_test(test_name: str, messages: list):
    print(f"\n{'='*50}")
    print(f"TEST: {test_name}")
    print(f"{'='*50}")
    
    print("\nINPUT HISTORY:")
    for m in messages:
        print(f"  [{m['role'].upper()}]: {m['content']}")

    graph = SHLGraphBuilder()
    
    print("\nEXECUTING GRAPH...")
    result = graph.invoke(messages)
    
    print("\nOUTPUT:")
    print(f"Reply: {result.get('reply')}")
    print(f"End of Conversation: {result.get('end_of_conversation')}")
    
    recs = result.get('recommendations', [])
    if recs:
        print(f"Recommendations ({len(recs)}):")
        for i, r in enumerate(recs, 1):
            name = r.name if hasattr(r, 'name') else r.get('name')
            test_type = r.test_type if hasattr(r, 'test_type') else r.get('test_type')
            print(f"  {i}. {name} [{test_type}]")
    else:
        print("Recommendations: None")
        
    return result


def main():
    # ---------------------------------------------------------
    # TEST 1: CONTEXT RETENTION FOR SHORT ANSWERS
    # ---------------------------------------------------------
    t1_msg = [
        {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
        {"role": "assistant", "content": "Sure. What is seniority level?"},
        {"role": "user", "content": "4 years"}
    ]
    run_test("1. Context Retention (Should combine 'Java developer' and '4 years')", t1_msg)

    # ---------------------------------------------------------
    # TEST 2: COMPARISON OF JOB ROLES
    # ---------------------------------------------------------
    t2_msg = [
        {"role": "user", "content": "give me diff between two job roles in their test explain it"}
    ]
    run_test("2. Job Role Comparison (Should hit Comparison Node)", t2_msg)

    # ---------------------------------------------------------
    # TEST 3: REFINE (History test)
    # ---------------------------------------------------------
    # Now user refines the previous recommendation
    t3_msg = t1_msg + [
        {"role": "assistant", "content": "Here are some assessments."},
        {"role": "user", "content": "Actually, drop the basic java tests and add something for personality fit."}
    ]
    res3 = run_test("3. Refinement with History (Should Update Shortlist)", t3_msg)

    # ---------------------------------------------------------
    # TEST 3: COMPARE
    # ---------------------------------------------------------
    t3_msg = [
        {"role": "user", "content": "What is the difference between OPQ32r and the Global Skills Assessment?"}
    ]
    run_test("3. Comparison", t3_msg)

    # ---------------------------------------------------------
    # TEST 4: REFUSE
    # ---------------------------------------------------------
    t4_msg = [
        {"role": "user", "content": "Can you give me legal advice on how to fire an employee?"}
    ]
    run_test("4. Refusal (Legal Advice)", t4_msg)

    # ---------------------------------------------------------
    # TEST 5: COMPLETION
    # ---------------------------------------------------------
    # Simulate a conversation that ends in satisfaction
    t5_msg = [
        {"role": "user", "content": "We need assessments for a sales manager."},
        {"role": "assistant", "content": "I recommend the Sales Transformation Report and OPQ MQ Sales Report."},
        {"role": "user", "content": "Perfect, let's go with those. Lock it in."}
    ]
    run_test("5. Completion / Satisfaction", t5_msg)


if __name__ == "__main__":
    main()
