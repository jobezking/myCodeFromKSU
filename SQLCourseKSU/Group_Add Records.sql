INSERT ALL
  INTO Insurance_T VALUES (1010,'Delta Dental','1809 Valley Place','Suite 5','Atlanta','GA',30005,8004563403)
  INTO Insurance_T VALUES (1020,'Aetna','543 Mint Dr','Suite 2','Alpharetta','MD',30097,8003456789)
  INTO Insurance_T VALUES (1030,'Cigna','987 Prune Pl', 'Suite 2k','Tampa','FL',30432, 8002134567)
  INTO Insurance_T VALUES (1040,'Guardian','1234 Heat Point','Suite 90','Baltimore','MD', 33456, 8889876543)
  INTO Insurance_T VALUES (1050,'Blue Cross','5050 Apple Rd','Box 1040','New York','NY', 10001, 8773862794)
SELECT * FROM dual;

INSERT ALL
  INTO Procedure_T VALUES (2393, 'Cavity', '180')
  INTO Procedure_T VALUES (0210, 'X-rays', '101')
  INTO Procedure_T VALUES (1110, 'Cleaning', '79')
  INTO Procedure_T VALUES (7210, 'Extraction', '214')
  INTO Procedure_T VALUES (2752, 'Crown', '990')
SELECT * FROM dual; 

INSERT ALL
  INTO Patient_T VALUES (002, 'Engrid', 'I', 'Pitts', '07-OCT-1983', '3456 Peak Walk', NULL, 'Buford', 'GA', 35180, 'Y', 1010, 112233445)
  INTO Patient_T VALUES (003, 'Nicholas', 'B', 'Jones', '19-SEP-1982', '987 Creek Point', 'Apt4A', 'Atlanta', 'GA', 30041, 'Y', 1020, 123456789) 
  INTO Patient_T VALUES (004, 'Job', 'L', 'King', '10-MAY-1986', '8567 Hollow Place', NULL, 'Chamblee', 'GA', 30341, 'Y', 1030, 234567891)
  INTO Patient_T VALUES (001, 'Sabina', 'R', 'Schmitt', '08-JUN-1984', '111 Rox Ln', 'Apt2', 'Duluth', 'GA', 30096, 'N', NULL, NULL)
  INTO Patient_T VALUES (005, 'Ronald', 'A', 'Smallings', '04-MAR-1982', '543 Far Road', 'Apt67', 'Lawrenceville', 'GA', 30045, 'N', NULL, NULL) 
SELECT * FROM dual; 

INSERT ALL
  INTO Treatment_T VALUES (123, 001, 2393, '10-JAN-2017', 'Proposed', 'Cavity', 'N') 
  INTO Treatment_T VALUES (456, 002, 0210, '10-FEB-2017', 'Performed', 'X-rays', 'Y')
  INTO Treatment_T VALUES (789, 003, 1110, '11-MAR-2017', 'Billed to Ins', 'Cleaning', 'Y')
  INTO Treatment_T VALUES (987, 004, 7210, '22-APR-2017', 'Billed to Ins', 'Extraction', 'Y')
  INTO Treatment_T VALUES (654, 005, 2752, '19-MAY-2017', 'Billed to Pt', 'Crown', 'N')
SELECT * FROM dual;
