from semantic_router import Route, RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(
    name="sentence-transformers/all-MiniLM-L6-v2"
)

# Route 1: FAQ / Text documents
faq = Route(
    name="faq",
    utterances=[
        # Vision & Mission
        "What is the vision of the institute?",
        "Tell me the mission of BNMIT.",
        "What are the vision and mission statements?",
        "Can you explain the college vision and mission?",

        # Faculty / Staff
        "Who are the faculty members?",
        "Tell me about the teaching staff",
        "Provide information about faculty achievements",
        "Who are the HODs and faculty members?",
        "who is head",
        "who is hod",

        # Placement guidance / Training
        "Give details about the training program.",
        "Do you have information about pre-placement training?",
        "Explain the placement process",
        "What training and development activities are conducted?",
        "Tell me about student career guidance",

        # Facilities / Student support
        "What are the college facilities?",
        "Tell me about students' corner",
        "What are the innovative teaching methods used?",
        "Provide information about the recruiters' corner",
        "Explain the message from HOD",
    ]
)

# Route 2: SQL / Placement statistics
sql = Route(
    name="sql",
    utterances=[
        # Average salary / package
        "What is the average salary for CSE in 2023?",
        "Average package for AIML students in 2024?",
        "What is the weighted average salary for ECE?",
        "Give the avg salary for all branches in 2023",

        # Total placements
        "How many AIML students got placed in 2022?",
        "Total placements for CSE in 2023",
        "Number of EEE students placed last year",
        "Show total placements by branch",

        # Company info
        "Which company offered the highest package?",
        "List the companies that visited in 2024",
        "Show companies hiring more than 10 students",
        "List all recruiters for CSE",

        # Year-wise / branch-wise queries
        "Show placements by year for Infosys",
        "Average package for AIML in 2022",
        "Total number of placements for ECE",
        "Highest salary offered to Mech branch students",
        "Total placements across all branches for 2023",
    ]
)


# Router setup
router = RouteLayer(routes=[faq, sql], encoder=encoder)

if __name__ == "__main__":
    print(router("Tell me about training and development activities").name)  # faq
    print(router("What is the highest package in 2023?").name)  # sql
    print(router("hav you had cofee").name)
