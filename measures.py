Total Sales = SUM(Sales[Amount])

Average Price = AVERAGE(Products[Price])


count
Number of Transactions = COUNT(Sales[TransactionID])
Number of Customers = COUNTA(Customers[CustomerName])


removing duplicates/distinct count
Unique Customers = DISTINCTCOUNT(Sales[CustomerID])


Sales in 2023 = CALCULATE(SUM(Sales[Amount]), Sales[Year] = 2023)


High Value Sales = CALCULATE(SUM(Sales[Amount]), FILTER(Sales, Sales[Amount] > 1000))

Total Revenue = SUMX(Sales, Sales[Quantity] * Sales[UnitPrice])


Sales Category = IF(Sales[Amount] > 1000, "High", "Low")


Profit Margin = DIVIDE(SUM(Sales[Profit]), SUM(Sales[Sales]), 0)


Sales YTD = TOTALYTD(SUM(Sales[Amount]), 'Date'[Date])

Sales LY = CALCULATE(SUM(Sales[Amount]), SAMEPERIODLASTYEAR('Date'[Date]))

% of Total Sales = 
DIVIDE(SUM(Sales[Amount]), CALCULATE(SUM(Sales[Amount]), ALL(Sales)))

Product Category = RELATED(Products[Category])

Total Revenue = SUMX(Sales, Sales[Quantity] * Sales[Price])


// Distinction Rate (Score ≥ 75)
Distinction Rate = 
DIVIDE(
    CALCULATE(COUNTROWS(Enrollments), Enrollments[ExamScore] >= 75),
    COUNTROWS(Enrollments)
)

// High Distinction Rate (Score ≥ 85)
High Distinction Rate = 
DIVIDE(
    CALCULATE(COUNTROWS(Enrollments), Enrollments[ExamScore] >= 85),
    COUNTROWS(Enrollments)
)

// Failure Rate (Score < 50)
Failure Rate = 
DIVIDE(
    CALCULATE(COUNTROWS(Enrollments), Enrollments[ExamScore] < 50),
    COUNTROWS(Enrollments)
)

// Average Attendance Rate
Avg Attendance = AVERAGE(Enrollments[AttendanceRate])

// Students with Perfect Attendance (≥ 90%)
Perfect Attendance Count = 
CALCULATE(COUNTROWS(Enrollments), Enrollments[AttendanceRate] >= 90)

// Average Tuition per Student
Avg Tuition = AVERAGE(Enrollments[TuitionFee])

// Average Scholarship per Student
Avg Scholarship = AVERAGE(Enrollments[ScholarshipAmount])

// Scholarship Ratio (what % of tuition is covered)
Scholarship Coverage Ratio = 
DIVIDE(SUM(Enrollments[ScholarshipAmount]), SUM(Enrollments[TuitionFee]))

// Students with Full Scholarship
Full Scholarship Count = 
CALCULATE(COUNTROWS(Enrollments), Enrollments[ScholarshipAmount] > 0)

// Students with No Scholarship
No Scholarship Count = 
CALCULATE(COUNTROWS(Enrollments), Enrollments[ScholarshipAmount] = 0)

// Total Revenue after Scholarships (already have Net Revenue)


// Completion Rate
Completion Rate = 
DIVIDE(
    CALCULATE(COUNTROWS(Enrollments), Enrollments[Status] = "Completed"),
    COUNTROWS(Enrollments)
)

// Current Active Students (In-Progress)
Active Enrollments = 
CALCULATE(COUNTROWS(Enrollments), Enrollments[Status] = "In-Progress")

// Failed Courses Count
Failed Courses = 
CALCULATE(COUNTROWS(Enrollments), Enrollments[Status] = "Failed")

// Average Credits Attempted
Avg Credits = AVERAGE(Enrollments[Credits])

// High Performers (Score ≥ 80 AND Attendance ≥ 85%)
High Performers Count = 
CALCULATE(
    COUNTROWS(Enrollments),
    Enrollments[ExamScore] >= 80,
    Enrollments[AttendanceRate] >= 85
)


// Total Students
Total Students = COUNTROWS('student')

// Gender Distribution
Male Students = CALCULATE(COUNTROWS('student'), 'student'[Gender] = "Male")
Female Students = CALCULATE(COUNTROWS('student'), 'student'[Gender] = "Female")

// Student Type Analysis
FullTime Students = CALCULATE(COUNTROWS('student'), 'student'[StudentType] = "Full-time")
PartTime Students = CALCULATE(COUNTROWS('student'), 'student'[StudentType] = "Part-time")

// Regional Distribution
Students by Region = COUNTROWS('student')  // Use with Region slicer

// Average Student Age
Avg Student Age = AVERAGE('student'[Age])




// Sponsorship Analysis
Sponsored Students = 
CALCULATE(COUNTROWS('student'), 'student'[SponsorshipType] <> "Self")

// Enrollments by Department (using relationships)
Department Enrollments = COUNTROWS(Enrollments)

// Department Performance
Department Pass Rate = 
DIVIDE(
    CALCALCULATE(COUNTROWS(Enrollments), Enrollments[ExamScore] >= 50),
    COUNTROWS(Enrollments)
)

// Department Revenue
Department Revenue = SUM(Enrollments[TuitionFee])

// Top Performing Department (by Avg Score)
Top Department Score = 
MAXX(
    VALUES(department[DepartmentName]),
    CALCULATE(AVERAGE(Enrollments[ExamScore]))
)


Students Without Scholarship = 
CALCULATE(COUNTROWS(Enrollments), Enrollments[ScholarshipAmount] = 0)

// Total Production Cost
Total Production Cost = SUMX('Production', 'Production'[Cost/Ton] * 'Production'[Quantity])

// Total Revenue
Total Revenue = SUMX('Production', 'Production'[Price/Ton] * 'Production'[Quantity])

// Profit
Profit = [Total Revenue] - [Total Production Cost]

// Profit Margin (%)
Profit Margin = DIVIDE([Profit], [Total Revenue], 0) * 100
