import requests

def main():
    print("\n" + "="*70)
    print("MEDICAL QUERY SYSTEM - Terminal Interface")
    print("="*70)
    print("Type 'exit' or 'quit' to stop\n")
    
    while True:
        # Get query from user
        query = input("Enter your query: ").strip()
        
        # Exit condition
        if query.lower() in ['exit', 'quit', '']:
            print("\n👋 Goodbye!\n")
            break
        
        try:
            # Send request
            response = requests.post(
                "http://localhost:8000/ask",
                json={"text": query},
                timeout=10
            )
            
            # Get result
            result = response.json()
            
            # Print answer
            print(f"\n{'─'*70}")
            print(f"ANSWER: {result['answer']}")
            print(f"{'─'*70}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()