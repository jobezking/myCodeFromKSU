CREATE TABLE Insurance_T 
(InsCompID      NUMBER(11,0)  NOT NULL, 
InsCompName     VARCHAR2(20), 
InsAdd1         VARCHAR2(20), 
InsAdd2         VARCHAR2(15), 
InsCity         VARCHAR2(20), 
InsState        VARCHAR2(2), 
InsZip          VARCHAR2(9), 
InsPhoneNum     NUMBER(11,0), 
CONSTRAINT Insurance_PK PRIMARY KEY (InsCompID)); 

CREATE TABLE Procedure_T 
(ProcedureID    INTEGER NOT NULL, 
ProcedureName   VARCHAR2(20), 
UnitCost        DECIMAL(6,2), 
CONSTRAINT Procedure_PK PRIMARY KEY (ProcedureID)); 

CREATE TABLE Patient_T 
(PatientNum     NUMBER(11,0)  NOT NULL, 
PatientFN       VARCHAR2(15),       
PatientMI       VARCHAR2(3), 
PatientLN       VARCHAR2(20), 
PatientDOB      DATE, 
PatientAdd1     VARCHAR2(20), 
PatientAdd2     VARCHAR2(15), 
PatientCity     VARCHAR2(20), 
PatientState    VARCHAR2(2), 
PatientZip      VARCHAR2(9), 
PatientInsStat  VARCHAR2(1) CONSTRAINT PatientInsStat_CK CHECK (PatientInsStat = 'Y' OR PatientInsStat = 'N'),
InsCompID       NUMBER (11,0), 
PatientInsID    NUMBER (11,0), 
CONSTRAINT Patient_PK PRIMARY KEY (PatientNum), 
CONSTRAINT Patient_FK FOREIGN KEY (InsCompID) REFERENCES Insurance_T (InsCompID),
CONSTRAINT InsCompID_CK CHECK (PatientInsStat = 'N' OR InsCompID IS NOT NULL),
CONSTRAINT PatientInsID_CK CHECK (PatientInsStat = 'N' OR PatientInsID IS NOT NULL));

CREATE TABLE Treatment_T 
(TxID           NUMBER(11,0)  NOT NULL, 
PatientNum      NUMBER(11,0)  NOT NULL, 
ProcedureID     INTEGER       NOT NULL, 
ProcedureDate   DATE, 
Status          VARCHAR2(20), 
Reason          VARCHAR2(20),
InsCoverage     VARCHAR2(1) CONSTRAINT InsCoverage_CK CHECK (InsCoverage = 'Y' OR InsCoverage = 'N'),
CONSTRAINT Treatment_PK PRIMARY KEY (TxID), 
CONSTRAINT Treatment_FK1 FOREIGN KEY (PatientNum) REFERENCES Patient_T (PatientNum), 
CONSTRAINT Treatment_FK2 FOREIGN KEY (ProcedureID) REFERENCES Procedure_T (ProcedureID));