🧩 Feature Engineering for Mixed-Type Variables
📘 About the Project

This project demonstrates practical feature engineering techniques to transform mixed-type variables — columns containing a blend of numeric and categorical data — into distinct, machine-learning–ready features.

In many real-world datasets, you’ll encounter columns that contain both numbers and letters, such as "A71", "B27", or "4". These are challenging to handle directly in machine learning pipelines, so this notebook walks through systematic ways to separate and encode them properly.

🎯 Project Focus

The notebook covers two major scenarios of mixed-type data:

Scenario	Description	Example
1. Numeric + Categorical in One Column	A column contains numeric values and categorical labels together.	'1', '4', 'A' in Travel_Partner
2. Alphanumeric Identifiers	A column combines letters and digits in identifiers.	'A71', 'B27', 'D57' in Ticket_ID
⚙️ Technologies Used

You only need essential Python data libraries:

import numpy as np
import pandas as pd

📊 Dataset Overview

The dataset used is mixed_variable.csv, which contains hypothetical travel or passenger data.
Below are the key columns used for demonstration:

Column Name	Example Values	Type	Mixed Component
Travel_Partner	'A', '2', '1', '4'	Mixed	Numerical digits + Alphabetical
Ticket_ID	'A71', 'B27', 'D57'	Mixed	Alphanumeric (Prefix + Suffix)
🧠 Technique 1: Separating Numeric and Categorical Data (Travel_Partner)

The Travel_Partner column contains both numeric-looking values ('1', '2', '3') and categorical identifiers ('A').
The goal is to split them into two new features.

🔹 Step 1: Extracting the Numerical Component (TP_Num)

We use pd.to_numeric() with errors='coerce' to convert all numeric-like strings to integers, while non-numeric values become NaN.

df['TP_Num'] = pd.to_numeric(df['Travel_Partner'], errors='coerce', downcast='integer')

🔹 Step 2: Extracting the Categorical Component (TP_Cat)

We use np.where() to label rows that were converted to NaN as categorical values.

df['TP_Cat'] = np.where(df['TP_Num'].isnull(), df['Travel_Partner'], np.nan)


✅ Result:

Travel_Partner	TP_Num	TP_Cat
'1'	1	NaN
'4'	4	NaN
'A'	NaN	'A'
⚙️ Technique 2: Splitting Alphanumeric Identifiers (Ticket_ID)

The Ticket_ID column holds values like 'A71' — a mix of categorical prefixes and numeric IDs.
We’ll split them into separate components for better model interpretability.

🔹 Step 1: Extracting the Numerical ID (ID_Num)

We use a regular expression to extract the numeric part.

df['ID_Num'] = df['Ticket_ID'].str.extract('(\d+)')

🔹 Step 2: Extracting the Categorical Prefix (ID_Cat)

The prefix (alphabetic character) is easily extracted using string indexing.

df['ID_Cat'] = df['Ticket_ID'].str[0]


✅ Result:

Ticket_ID	ID_Cat	ID_Num
'A71'	'A'	71
'B27'	'B'	27
'D57'	'D'	57
🧾 Summary Table
Original Column	Technique Used	New Features	Purpose
Travel_Partner	pd.to_numeric, np.where	TP_Num, TP_Cat	Separate numeric and categorical data
Ticket_ID	str.extract, str[0]	ID_Num, ID_Cat	Split alphanumeric identifiers
💡 Key Takeaways

Always inspect your column types — mixed-type variables can silently affect model performance.

Regex and Pandas string methods are powerful for parsing alphanumeric data.

By separating components, you can treat numeric and categorical parts independently, improving model interpretability and accuracy.

🧰 Export / Integration

Once cleaned, the processed DataFrame can easily be exported or visualized:

df.to_csv('cleaned_mixed_variable.csv', index=False)

🚀 Author

Anshuman [@Machine Learning Series]
Exploring patterns in data, one feature at a time.
