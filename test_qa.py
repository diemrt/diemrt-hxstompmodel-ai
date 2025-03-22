from hx_stomp_qa import HXStompQA

def main():
    print("Line 6 HX Stomp QA System")
    print("-------------------------")
    print("Type 'quit' to exit\n")

    qa_system = HXStompQA()

    while True:
        question = input("\nAsk a question about HX Stomp presets: ")
        
        if question.lower() == 'quit':
            break
            
        try:
            result = qa_system.answer_question(question)
            print("\nAnswer:", result['answer'])
            print("\nContext used:", result['context'][:200] + "..." if len(result['context']) > 200 else result['context'])
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()