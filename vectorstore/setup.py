import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

def create_all_chunks():
    docs = []

    # --- SECTION 1: Team Profiles ---
    teams = [
        ("Mumbai Indians", "MI", "Wankhede Stadium", "Hardik Pandya", "Mark Boucher", 5, "5th"),
        ("Chennai Super Kings", "CSK", "MA Chidambaram Stadium", "Ruturaj Gaikwad", "Stephen Fleming", 5, "Runner-up"),
        ("Royal Challengers Bengaluru", "RCB", "M Chinnaswamy Stadium", "Faf du Plessis", "Andy Flower", 3, "Champions"),
        ("Kolkata Knight Riders", "KKR", "Eden Gardens", "Shreyas Iyer", "Chandrakant Pandit", 3, "Champions"),
        ("Delhi Capitals", "DC", "Arun Jaitley Stadium", "Rishabh Pant", "Ricky Ponting", 0, "6th"),
        ("Punjab Kings", "PBKS", "IS Bindra Stadium", "Shikhar Dhawan", "Trevor Bayliss", 0, "9th"),
        ("Rajasthan Royals", "RR", "Sawai Mansingh Stadium", "Sanju Samson", "Kumar Sangakkara", 2, "Runner-up"),
        ("Sunrisers Hyderabad", "SRH", "Rajiv Gandhi Intl Stadium", "Pat Cummins", "Daniel Vettori", 1, "4th"),
        ("Lucknow Super Giants", "LSG", "BRSABV Ekana Stadium", "KL Rahul", "Justin Langer", 0, "7th"),
        ("Gujarat Titans", "GT", "Narendra Modi Stadium", "Shubman Gill", "Ashish Nehra", 2, "8th"),
    ]
    for name, short, venue, captain, coach, titles, pos in teams:
        docs.append(Document(
            page_content=f"{name} ({short}): home ground {venue}, captain {captain}, coach {coach}, {titles} IPL titles, 2024 position {pos}.",
            metadata={"section": "team", "team_name": name, "short": short, "season": 2024}
        ))

    # --- SECTION 2: Batting Stats ---
    batting = [
        ("Virat Kohli", "RCB", 237, 7263, 37.17, 130.0, 7, 50, "113", "Top-order bat"),
        ("Rohit Sharma", "MI", 243, 6211, 29.57, 130.5, 1, 40, "109*", "Opener"),
        ("Shubman Gill", "GT", 98, 3196, 42.61, 148.7, 3, 24, "129", "Opener"),
        ("Ruturaj Gaikwad", "CSK", 87, 2835, 38.83, 136.8, 2, 19, "101*", "Opener"),
        ("Sanju Samson", "RR", 171, 4410, 29.86, 140.5, 4, 28, "119", "WK-Bat"),
        ("KL Rahul", "LSG", 115, 4163, 47.30, 134.3, 4, 33, "132*", "Opener/WK"),
        ("Shreyas Iyer", "KKR", 115, 3122, 33.57, 127.8, 1, 25, "96", "Middle-order"),
        ("Rishabh Pant", "DC", 111, 3284, 35.31, 148.3, 0, 18, "128*", "WK-Bat"),
        ("Hardik Pandya", "MI", 133, 2754, 30.60, 145.5, 0, 15, "91*", "All-rounder"),
        ("Suryakumar Yadav", "MI", 148, 3225, 32.57, 161.7, 0, 23, "103", "Middle-order"),
        ("David Warner", "DC", 184, 6565, 41.42, 139.9, 4, 59, "126", "Opener"),
        ("Jos Buttler", "RR", 89, 3422, 46.24, 149.2, 5, 21, "124", "Opener/WK"),
        ("Faf du Plessis", "RCB", 121, 3779, 38.56, 134.2, 2, 30, "96", "Opener"),
        ("Travis Head", "SRH", 21, 872, 43.60, 191.6, 1, 6, "102", "Opener"),
        ("Abhishek Sharma", "SRH", 40, 1254, 33.78, 188.3, 1, 7, "135*", "Opener"),
        ("MS Dhoni", "CSK", 250, 5243, 39.42, 135.9, 0, 24, "84*", "WK-Finisher"),
        ("AB de Villiers", "RCB", 184, 5162, 39.70, 151.7, 3, 38, "133*", "Middle-order"),
        ("Ishan Kishan", "MI", 105, 2644, 27.28, 136.2, 1, 15, "99", "Opener/WK"),
        ("Quinton de Kock", "LSG", 78, 2516, 35.43, 138.9, 1, 18, "140*", "Opener/WK"),
        ("Heinrich Klaasen", "SRH", 54, 1892, 43.95, 162.8, 1, 13, "104", "Middle-order"),
    ]
    for player, team, mat, runs, avg, sr, hundreds, fifties, hs, role in batting:
        docs.append(Document(
            page_content=f"{player} batting: {mat} matches, {runs} runs, average {avg}, strike rate {sr}, {hundreds} centuries, {fifties} fifties, highest score {hs}, role {role}.",
            metadata={"section": "batting", "player_name": player, "team": team, "role": role}
        ))

    # --- SECTION 3: Bowling Stats ---
    bowling = [
        ("Yuzvendra Chahal", "RR", 155, 205, 22.64, 7.63, 17.8, "5/40", "Leg-spin"),
        ("DJ Bravo", "CSK", 161, 183, 25.05, 8.43, 17.7, "4/22", "Medium-fast"),
        ("Lasith Malinga", "MI", 122, 170, 19.80, 7.14, 16.6, "5/13", "Yorker specialist"),
        ("Jasprit Bumrah", "MI", 135, 170, 22.50, 7.39, 18.2, "5/10", "Pace/Yorker"),
        ("Amit Mishra", "DC", 154, 166, 24.34, 7.36, 19.8, "5/17", "Leg-spin"),
        ("Sunil Narine", "KKR", 174, 163, 24.09, 6.69, 21.5, "5/19", "Mystery off-spin"),
        ("Harbhajan Singh", "MI", 163, 150, 26.39, 6.97, 22.7, "5/18", "Off-spin"),
        ("Kagiso Rabada", "DC", 74, 113, 21.74, 8.34, 15.6, "4/21", "Fast bowling"),
        ("Trent Boult", "RR", 78, 105, 23.43, 8.18, 17.1, "4/18", "Swing bowling"),
        ("Mohammed Shami", "GT", 88, 114, 22.38, 8.02, 16.7, "4/16", "Fast bowling"),
        ("Pat Cummins", "SRH", 57, 78, 25.17, 8.45, 17.8, "4/14", "Fast bowling"),
        ("Varun Chakravarthy", "KKR", 67, 102, 22.78, 7.11, 19.2, "5/20", "Mystery spin"),
        ("Rashid Khan", "GT", 92, 131, 20.44, 6.68, 18.3, "5/17", "Leg-spin"),
    ]
    for player, team, mat, wkts, avg, econ, sr, best, bowl_type in bowling:
        docs.append(Document(
            page_content=f"{player} bowling: {mat} matches, {wkts} wickets, average {avg}, economy {econ}, strike rate {sr}, best figures {best}, type {bowl_type}.",
            metadata={"section": "bowling", "player_name": player, "team": team, "bowl_type": bowl_type}
        ))

    # --- SECTION 4: Head-to-Head Records ---
    h2h = [
        ("MI", "CSK", 35, 20, 14, "MI,CSK,MI,CSK,MI", "235/1 MI", "Bumrah vs Dhoni"),
        ("RCB", "KKR", 32, 14, 18, "KKR,RCB,KKR,KKR,RCB", "213/2 RCB", "Kohli vs Narine"),
        ("CSK", "RCB", 30, 22, 8, "CSK,CSK,RCB,CSK,RCB", "226/6 RCB", "Spinners vs Power"),
        ("MI", "SRH", 28, 16, 12, "SRH,MI,SRH,MI,SRH", "219/4 SRH", "Bumrah vs Head"),
        ("RR", "DC", 26, 14, 12, "RR,DC,RR,RR,DC", "217/5 RR", "Buttler vs Rabada"),
        ("KKR", "MI", 34, 17, 17, "MI,KKR,KKR,MI,KKR", "232/2 KKR", "Narine vs Bumrah"),
        ("SRH", "RCB", 22, 11, 11, "SRH,RCB,SRH,RCB,SRH", "287/3 SRH", "Head vs Kohli"),
        ("PBKS", "RR", 28, 14, 14, "RR,PBKS,RR,RR,PBKS", "221/3 PBKS", "Dhawan vs Samson"),
        ("DC", "CSK", 30, 14, 16, "CSK,DC,CSK,CSK,DC", "208/4 DC", "Pant vs Dhoni"),
    ]
    for t1, t2, total, t1w, t2w, last5, highscore, keyfactor in h2h:
        docs.append(Document(
            page_content=f"{t1} vs {t2}: {total} total matches, {t1} won {t1w}, {t2} won {t2w}. Last 5 results: {last5}. High score: {highscore}. Key factor: {keyfactor}.",
            metadata={"section": "h2h", "team1": t1, "team2": t2}
        ))

    # --- SECTION 5: Venue Reports ---
    venues = [
        ("Wankhede Stadium", "Mumbai", "flat", 175, "Batting-friendly", "High evening",
         "Bat first, use pace death bowling. Dew makes chasing easy in evening. Bumrah's yorkers nearly unplayable here. Teams batting first win 45%."),
        ("MA Chidambaram Stadium", "Chennai", "slow turning", 155, "Bowling-friendly", "Low",
         "Most spin-friendly pitch in IPL. CSK wins 65% of home games. Bowl first wins 60%. Spinners unplayable in second innings."),
        ("M Chinnaswamy Stadium", "Bengaluru", "flat short boundary", 185, "Very bat-friendly", "Moderate",
         "Most batter-friendly ground. 200+ totals common. No total is safe. Kohli averages 58 here. Multiple 220+ scores in 2024."),
        ("Eden Gardens", "Kolkata", "good pace bounce", 165, "Balanced", "High evening",
         "Flexible strategy. Both spinners and pacers effective. KKR mystery spinners excel here. Good pitch for all-rounders."),
        ("Narendra Modi Stadium", "Ahmedabad", "flat big ground", 170, "Slightly batting", "Low",
         "Big ground limits sixes. Bat first. Spinners effective in second half. GT's home advantage significant."),
        ("Rajiv Gandhi Intl Stadium", "Hyderabad", "flat bouncy", 178, "Batting-friendly", "High",
         "SRH scored 250+ twice in 2024. Heavy dew makes chasing almost a formality in night matches. Bowl first always correct call."),
        ("Sawai Mansingh Stadium", "Jaipur", "slow low", 158, "Bowling-friendly", "Low",
         "Bowl first. Spinners dominate afternoon games. Low and slow surface. RR's home spinners are lethal here."),
        ("IS Bindra Stadium", "Mohali", "good carry seam", 168, "Balanced", "Moderate",
         "Flexible. Fast bowlers get swing early. Good carry for pace. Both batting and bowling teams have chance."),
    ]
    for vname, city, pitch, avg1st, nature, dew, narrative in venues:
        docs.append(Document(
            page_content=f"{vname} in {city}: {pitch} pitch, average first innings {avg1st}, {nature}, dew factor {dew}. {narrative}",
            metadata={"section": "venue", "venue_name": vname, "city": city, "pitch_type": pitch}
        ))

    # --- SECTION 6: Season-wise Performance ---
    season_data = [
        ("MI", [("2019","Champions"),("2020","Champions"),("2021","5th"),("2022","5th"),("2023","5th"),("2024","5th")]),
        ("CSK", [("2019","Runner-up"),("2020","Runner-up"),("2021","Champions"),("2022","9th"),("2023","Champions"),("2024","Runner-up")]),
        ("KKR", [("2019","5th"),("2020","6th"),("2021","2nd"),("2022","7th"),("2023","4th"),("2024","Champions")]),
        ("RCB", [("2019","8th"),("2020","4th"),("2021","Runner-up"),("2022","8th"),("2023","2nd"),("2024","Champions")]),
        ("DC",  [("2019","3rd"),("2020","Runner-up"),("2021","3rd"),("2022","4th"),("2023","6th"),("2024","6th")]),
        ("RR",  [("2019","7th"),("2020","8th"),("2022","Runner-up"),("2023","Runner-up"),("2024","Runner-up")]),
        ("SRH", [("2019","6th"),("2020","3rd"),("2022","8th"),("2023","8th"),("2024","4th")]),
        ("GT",  [("2022","Champions"),("2023","Runner-up"),("2024","8th")]),
        ("LSG", [("2022","3rd"),("2023","3rd"),("2024","7th")]),
        ("PBKS",[("2019","6th"),("2020","6th"),("2021","6th"),("2022","6th"),("2023","7th"),("2024","9th")]),
    ]
    for team, seasons in season_data:
        for year, position in seasons:
            docs.append(Document(
                page_content=f"{team} finished {position} in IPL {year} season.",
                metadata={"section": "season", "team": team, "year": int(year)}
            ))

    # --- SECTION 7: Recent Form ---
    form_data = [
        ("Virat Kohli", "RCB", [92,47,0,73,113], "Excellent", 65.0),
        ("Rohit Sharma", "MI", [11,8,34,67,43], "Average", 32.6),
        ("Travis Head", "SRH", [102,34,58,12,89], "Excellent", 59.0),
        ("Jos Buttler", "RR", [67,0,89,45,32], "Mixed", 46.6),
        ("Hardik Pandya", "MI", [34,12,0,56,28], "Poor", 26.0),
        ("Ruturaj Gaikwad", "CSK", [78,34,12,90,56], "Good", 54.0),
        ("Abhishek Sharma", "SRH", [135,43,67,8,92], "Excellent", 69.0),
        ("KL Rahul", "LSG", [45,67,23,12,54], "Moderate", 40.2),
    ]
    for player, team, scores, trend, avg in form_data:
        docs.append(Document(
            page_content=f"{player} recent form last 5 matches: {scores}. Trend: {trend}. Average last 5: {avg}.",
            metadata={"section": "form", "player_name": player, "team": team, "season": 2024, "trend": trend}
        ))

    # Bowling form
    docs.append(Document(
        page_content="Jasprit Bumrah recent form last 5 matches: 3/24, 2/18, 1/32, 4/10, 2/28. Trend: Excellent. Average 2.4 wickets per game.",
        metadata={"section": "form", "player_name": "Jasprit Bumrah", "team": "MI", "season": 2024, "trend": "Excellent"}
    ))
    docs.append(Document(
        page_content="Rashid Khan recent form last 5 matches: 2/18, 1/22, 3/15, 0/34, 2/20. Trend: Consistent. Average 1.6 wickets per game.",
        metadata={"section": "form", "player_name": "Rashid Khan", "team": "GT", "season": 2024, "trend": "Consistent"}
    ))

    # --- SECTION 8: Records ---
    records = [
        ("Highest Team Score", "287/3", "SRH", "vs RCB, Uppal Hyderabad, 2024"),
        ("Highest Individual Score", "175* off 66 balls", "Chris Gayle", "vs PWI, Bengaluru, 2013"),
        ("Most Runs Career", "7263", "Virat Kohli", "2008-2024"),
        ("Most Wickets Career", "205", "Yuzvendra Chahal", "2011-2024"),
        ("Most Centuries", "7", "Virat Kohli", "2008-2024"),
        ("Best Bowling Figures", "6/12", "Alzarri Joseph", "vs SRH, Wankhede, 2019"),
        ("Most Sixes Career", "357", "Chris Gayle", "2009-2022"),
        ("Fastest Fifty", "14 balls", "KL Rahul", "vs KXIP, Dubai, 2020"),
        ("Most Titles", "5", "MI and CSK", "various years"),
        ("Most Matches Player", "250", "MS Dhoni", "2008-2024"),
        ("Highest Chase", "232/2", "KKR", "vs RCB, Eden Gardens, 2024"),
        ("Most Runs Single Season", "973", "Virat Kohli", "2016 season"),
        ("Lowest Total", "49 all out", "RCB", "vs KKR, Eden Gardens, 2017"),
    ]
    for category, record, holder, context in records:
        docs.append(Document(
            page_content=f"IPL Record — {category}: {record} by {holder} ({context}).",
            metadata={"section": "records", "category": category}
        ))

    # --- SECTION 11: Conflicting Data (keep BOTH versions) ---
    docs.append(Document(
        page_content="Virat Kohli career IPL runs: 7263 (primary source, official IPL data).",
        metadata={"section": "records", "player_name": "Virat Kohli", "source": "primary", "conflict": True}
    ))
    docs.append(Document(
        page_content="Virat Kohli career IPL runs: 7084 (secondary source, unofficial stats portal).",
        metadata={"section": "records", "player_name": "Virat Kohli", "source": "secondary", "conflict": True}
    ))
    docs.append(Document(
        page_content="Yuzvendra Chahal career IPL wickets: 205 (primary source, official IPL data).",
        metadata={"section": "records", "player_name": "Yuzvendra Chahal", "source": "primary", "conflict": True}
    ))
    docs.append(Document(
        page_content="Yuzvendra Chahal career IPL wickets: 187 (secondary source, possibly outdated dataset).",
        metadata={"section": "records", "player_name": "Yuzvendra Chahal", "source": "secondary", "conflict": True}
    ))

    return docs


def build_vectorstore():
    print("Building vector store...")
    docs = create_all_chunks()
    print(f"Total chunks created: {len(docs)}")

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="ipl_rag"
    )
    print("Vector store built and saved to ./chroma_db")
    return vectorstore


def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="ipl_rag"
    )