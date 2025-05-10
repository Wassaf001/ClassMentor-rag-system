import requests
import json
import time
import asyncio
import concurrent.futures
from tabulate import tabulate

async def query_server(url, query, method_name):
    """Query a RAG server and return the response with timing information"""
    headers = {"Content-Type": "application/json"}
    data = {"query": query}
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                "method": method_name,
                "status": "Success",
                "response": result["response"],
                "time": elapsed,
                "time_reported": result.get("response_time", "N/A"),
                "doc_count": len(result.get("relevant_documents", [])),
                "full_response": result
            }
        else:
            return {
                "method": method_name,
                "status": f"Error: {response.status_code}",
                "response": None,
                "time": elapsed,
                "time_reported": "N/A",
                "doc_count": 0,
                "full_response": None
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "method": method_name,
            "status": f"Exception: {str(e)}",
            "response": None,
            "time": elapsed,
            "time_reported": "N/A",
            "doc_count": 0,
            "full_response": None
        }

async def query_all_servers(query):
    """Query all three RAG servers and return their responses"""
    servers = [
        {"url": "http://0.0.0.0:8000/query", "name": "Standard RAG"},
        # {"url": "http://0.0.0.0:8001/query", "name": "RAG Fusion"},
        # {"url": "http://0.0.0.0:8002/query", "name": "TF-IDF RAG"}
    ]
    
    tasks = [query_server(server["url"], query, server["name"]) for server in servers]
    
    results = await asyncio.gather(*tasks)
    return results

def display_results(results):
    """Display the results in a formatted way"""
    table_data = []
    for result in results:
        table_data.append([
            result["method"],
            result["status"],
            f"{result['time']:.2f}s",
            result["time_reported"],
            result["doc_count"]
        ])
    
    print("\n" + "="*80)
    print("SUMMARY OF RAG APPROACHES")
    print("="*80)
    print(tabulate(
        table_data,
        headers=["Method", "Status", "Total Time", "LLM Time", "Doc Count"],
        tablefmt="grid"
    ))
    
    for result in results:
        if result["response"]:
            print("\n" + "="*80)
            print(f"RESPONSE FROM: {result['method']}")
            print("="*80)
            print(result["response"])
            
    with open("rag_comparison_results.json", "w") as f:
        json_results = {r["method"]: r["full_response"] for r in results if r["full_response"]}
        json.dump(json_results, f, indent=2)
    print("\nFull results saved to rag_comparison_results.json")

def main():
    # query = input("Enter your query: ")
    query = "Objective: Generate a comprehensive, accurate, and exceptionally detailed answer for the following examination question(s) with the goal to produce an answer that would achieve full marks according to a rigorous academic evaluation. Input Question(s): Solve the previous year questions of Mobile Computing for the year 2023-2024. Output Requirements & Formatting Guidelines: Completeness & Accuracy: Address every part of the question thoroughly without missing any sub-parts or implicit requirements, ensuring all information provided is factually correct and up-to-date according to established knowledge in the subject, and define all key terms and concepts relevant to the question before using them or as part of their first use. Depth of Explanation: Explain concepts from fundamental principles, not assuming prior advanced knowledge beyond what is typically expected for the specified examination level, provide step-by-step derivations for any formulas or mathematical results explaining the logic and assumptions at each step, include relevant examples, applications, or implications where appropriate to demonstrate a deeper understanding, and if applicable, discuss any limitations, assumptions, or special conditions related to the concepts or solutions. Structure & Clarity: Organize the answer logically with clear headings and subheadings if the question is multi-faceted, use clear, concise, and unambiguous language maintaining a formal and academic tone, and employ bullet points or numbered lists for distinct points, advantages, disadvantages, steps, etc., to enhance readability, although for this input method, simulate these with clear phrasing. Diagrams & Figures: Crucially, where a diagram, figure, graph, or schematic would significantly aid in explanation or is explicitly or implicitly required by the question, you MUST indicate this by using a placeholder like '[[DIAGRAM: Description]]' or '[[FIGURE: Description]]' or '[[GRAPH: Description]]' followed by a detailed description of what the diagram should depict clear enough for a human to draw it accurately including key components, labels, relationships, directions for vectors, and the overall message, referencing the placeholder in your text where appropriate from the book yi bing li wireless and mobile network architectures.pdf. Mathematical Notation & Units: Use standard mathematical notation consistently, clearly define any variables used, and ensure all numerical answers or physical quantities are expressed with appropriate SI units or other specified units. Marking Scheme Considerations (Simulated): Imagine the question carries a typical mark, for example '10 marks' or '15 marks', and allocate your effort and detail proportionally, ensuring that if the question asks for a specific number of points you provide exactly that number explained thoroughly. Tone & Style: Answer as an expert in the field aiming to educate and demonstrate complete mastery, avoiding colloquialisms or overly casual language. Pre-computation/Pre-analysis: Mentally identify core concepts, break down the question, outline your answer structure, and determine where diagrams are essential. Example for diagram handling: For explaining a P-N junction diode under forward bias, you might say '...This is illustrated in the diagram below. [[DIAGRAM: Circuit diagram of a forward-biased P-N junction diode, showing P-type and N-type regions, depletion layer, external DC voltage source with correct polarity, optional series resistor, arrows for conventional current and electron flow, and reduced depletion region width.]] The diagram shows that the applied forward voltage opposes the built-in potential barrier...' Also, give reference to page no. of book for diagrams. Please proceed to generate the answer for the provided question(s) following all these guidelines meticulously."
    print(f"\nQuerying all RAG servers with: '{query}'")
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(query_all_servers(query))
    display_results(results)

if __name__ == "__main__":
    main()