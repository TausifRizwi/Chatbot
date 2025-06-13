# subject_agent.py

class SubjectAgent:
    """
    An agent designed to answer questions within a specific subject area,
    combining predefined exact Q&A pairs with a local keyword-based search
    within larger textual content (no external API calls).
    """

    def __init__(self, subject_name: str):
        """
        Initializes the SubjectAgent with a specific subject and its knowledge base.

        Args:
            subject_name (str): The name of the subject this agent specializes in.
        """
        self.subject_name = subject_name
        self.qa_pairs, self.long_texts = self._load_knowledge_base()
        print(f"[{self.subject_name} Agent]: Initialized with internal knowledge base (Q&A + long texts).")

    def _load_knowledge_base(self) -> tuple[dict, list]:
        """
        Loads a predefined set of questions-answers and longer textual content
        for the agent's subject. This simulates a larger local knowledge base.

        Returns:
            tuple[dict, list]: A tuple containing a dictionary of Q&A pairs
                               and a list of longer text strings.
        """
        qa_pairs = {}
        long_texts = []

        # You can expand these significantly!
        if self.subject_name == "Polity":
            qa_pairs = {
                "what is the capital of india?": "The capital of India is New Delhi.",
                "who is the current prime minister of india?": "The current Prime Minister of India is Narendra Modi.",
                "what is the supreme law of india?": "The Constitution of India is the supreme law of India.",
                "what are fundamental rights?": "Fundamental Rights are a group of rights recognized by the Constitution of India as basic human freedoms that every Indian citizen has the right to enjoy. They are enshrined in Part III of the Constitution.",
                "how many articles are there in indian constitution?": "As of latest amendments, the Indian Constitution has 488 articles in 25 parts and 12 schedules.",
                "what is a republic?": "A republic is a form of government in which the country is considered a 'public matter', not the private concern or property of a single ruler. The head of state is not a monarch and is usually elected.",
                "what is the preamble of indian constitution?": "The Preamble to the Constitution of India is an introductory statement that sets out the guiding purpose and principles of the document, and it indicates the source of the document's authority, and the people to whom it is bestowed.",
                "who is the president of india?": "The President of India is the head of state of the Republic of India. The President is the ceremonial head of the executive, legislature and judiciary of India and is the commander-in-chief of the Indian Armed Forces.",
                "what is the role of election commission of india?": "The Election Commission of India (ECI) is an autonomous constitutional body responsible for administering election processes in India at national and state levels."
            }
            long_texts = [
                """The Constitution of India is the longest written constitution of any sovereign country in the world. It was adopted by the Constituent Assembly of India on 26 November 1949 and became effective on 26 January 1950. The Constitution provides for a parliamentary form of government, which is federal in structure with certain unitary features. It establishes a secular democratic republic, guaranteeing justice, liberty, equality to its citizens and promoting fraternity among them. India operates under a 'Cabinet system' of government with the Prime Minister at its head.""",
                """Fundamental Rights are a cornerstone of the Indian Constitution, enshrined in Part III. They include the Right to Equality, Right to Freedom, Right against Exploitation, Right to Freedom of Religion, Cultural and Educational Rights, and Right to Constitutional Remedies. These rights are justiciable, meaning citizens can approach the Supreme Court or High Courts if their fundamental rights are violated. However, these rights are not absolute and are subject to reasonable restrictions.""",
                """Directive Principles of State Policy (DPSP) are enumerated in Part IV of the Constitution. Unlike Fundamental Rights, DPSP are not justiciable, meaning they cannot be enforced by any court. However, they are fundamental in the governance of the country and it shall be the duty of the State to apply these principles in making laws. They aim at establishing a welfare state and achieving social and economic justice. Examples include promoting the welfare of the people, securing a social order, and providing free legal legal aid.""",
                """The Indian Parliament is a bicameral legislature consisting of the President and two Houses: the Lok Sabha (House of the People) and the Rajya Sabha (Council of States). The Lok Sabha members are directly elected by the people, while the Rajya Sabha members are indirectly elected. The primary functions of Parliament include making laws, controlling the executive, overseeing public finance, and serving as a forum for debate on public issues."""
            ]
        elif self.subject_name == "History":
            qa_pairs = {
                "when did india gain independence?": "India gained independence on August 15, 1947.",
                "who was the first prime minister of india?": "Jawaharlal Nehru was the first Prime Minister of India.",
                "when was the battle of panipat fought?": "The First Battle of Panipat was fought in 1526.",
                "who built the taj mahal?": "The Taj Mahal was built by the Mughal emperor Shah Jahan.",
                "what was the jallianwala bagh massacre?": "The Jallianwala Bagh Massacre was a horrific incident that took place on April 13, 1919, in Amritsar, Punjab, where British Indian Army soldiers, under the command of Colonel Reginald Dyer, fired machine guns into a crowd of unarmed Indian civilians."
            }
            long_texts = [
                """The ancient history of India is vast, spanning from the Indus Valley Civilization (c. 2500-1900 BCE) to the rise of major empires like the Mauryas and Guptas. The Mauryan Empire, founded by Chandragupta Maurya, was significant for its vast territorial extent and the reign of Emperor Ashoka, who promoted Buddhism. The Gupta Empire is often referred to as the 'Golden Age of India' due to significant advancements in science, mathematics, astronomy, art, and philosophy.""",
                """The medieval period in India saw the rise of various regional kingdoms and the establishment of the Delhi Sultanate, followed by the powerful Mughal Empire. The Mughals, starting with Babur in 1526, unified much of the Indian subcontinent and brought about a blend of Persian, Central Asian, and Indian cultures, evident in their architecture, art, and administration. The decline of the Mughal Empire paved the way for the rise of regional powers and later, European colonial expansion.""",
                """Modern Indian history is largely defined by the British colonial period, beginning with the East India Company's influence and culminating in direct British Raj. The struggle for independence intensified in the early 20th century, led by figures like Mahatma Gandhi, Jawaharlal Nehru, and Sardar Patel, employing various methods including non-violent civil disobedience. This period concluded with India's partition and independence in 1947.""",
                """The Quit India Movement was a significant civil disobedience movement launched in August 1942 by Mahatma Gandhi. It demanded an end to British rule in India. Though the movement faced severe repression and many leaders were arrested, it played a crucial role in galvanizing the Indian population and demonstrating widespread desire for independence. It was a major turning point in India's freedom struggle."""
            ]
        elif self.subject_name == "Economics":
            qa_pairs = {
                "what is gdp?": "GDP stands for Gross Domestic Product. It is the total monetary or market value of all the finished goods and services produced within a country's borders in a specific time period.",
                "what is inflation?": "Inflation refers to the rate at which the general level of prices for goods and services is rising, and subsequently, purchasing power is falling.",
                "what is a recession?": "A recession is a significant decline in economic activity spread across the economy, lasting more than a few months, normally visible in real GDP, real income, employment, industrial production, and wholesale-retail sales.",
                "what is fiscal policy?": "Fiscal policy refers to the use of government spending and taxation to influence the economy. It is primarily determined by the executive and legislative branches of the government.",
                "what is monetary policy?": "Monetary policy refers to the actions undertaken by a central bank (like the RBI in India) to influence the availability and cost of money and credit to help promote national economic goals.",
                "what is a budget deficit?": "A budget deficit occurs when government expenditures exceed revenues, indicating a financial shortfall."
            }
            long_texts = [
                """India's economy is a developing mixed economy. It is the world's fifth-largest economy by nominal GDP and the third-largest by purchasing power parity (PPP). Key sectors include agriculture, manufacturing, and services. The services sector, particularly IT and business process outsourcing, has been a major driver of economic growth in recent decades. The Indian economy faces challenges such as income inequality, poverty, and infrastructure gaps, but also presents significant opportunities for growth and development.""",
                """The Reserve Bank of India (RBI) is the central bank of India. It was established on April 1, 1935, and is responsible for regulating the monetary policy of the Indian rupee. Its primary functions include issuing currency, acting as a banker to the government, managing foreign exchange, and regulating the banking system. The RBI plays a crucial role in maintaining financial stability and promoting economic growth in the country.""",
                """Public finance in India deals with the management of government revenue, expenditures, and debt. It involves understanding how the government raises money (e.g., through taxes, borrowings) and how it spends that money (e.g., on infrastructure, defense, social welfare). The Union Budget, presented annually, is a key document that outlines the government's financial plans for the upcoming fiscal year."""
            ]
        return qa_pairs, long_texts

    def answer_question(self, question: str) -> str:
        """
        Provides an answer to a given question. It first attempts an exact match
        in its Q&A knowledge base. If not found, it performs a keyword search
        within its longer textual content to provide a relevant paragraph.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The answer found, or a relevant text snippet, or a default message.
        """
        print(f"[{self.subject_name} Agent]: Attempting to answer question: '{question}'...")
        normalized_question = question.strip().lower()

        # 1. Try exact Q&A match first
        exact_answer = self.qa_pairs.get(normalized_question)
        if exact_answer:
            return exact_answer

        # 2. If no exact match, perform keyword search on long texts
        # Split the question into individual words to use as keywords
        keywords = normalized_question.split()
        relevant_snippets = []

        for text in self.long_texts:
            # Check if all keywords are present in the current text snippet
            # This is a simple 'AND' logic for keyword matching
            if all(keyword in text.lower() for keyword in keywords):
                relevant_snippets.append(text)

        if relevant_snippets:
            # If relevant snippets are found, combine them and return
            # Limiting to the first 2 snippets for brevity in the output
            return ("Based on my knowledge:\n\n" +
                    "\n\n".join(relevant_snippets[:2]) +
                    "\n\n(This answer is derived from relevant sections of my knowledge base.)")
        else:
            # If no exact match and no relevant snippets found
            return (f"I'm sorry, I don't have enough information in my {self.subject_name} "
                    "knowledge base to answer that specific question precisely, "
                    "nor could I find directly relevant text for keywords. "
                    "Please try rephrasing, or asking one of my known questions.")