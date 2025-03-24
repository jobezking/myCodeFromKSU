select unitcost,procedurename from procedure_t order by unitcost desc;
alter table insurance_t rename column InsPhoneNum To InsPhoneNmbr;
alter table patient_t add (PatientPhoneNum Number(11,0) DEFAULT 55512345678);
SELECT AVG(UnitCost) "Average Procedure Cost" FROM Procedure_T;
SELECT SUM(UnitCost) "Sum Of Procedure Costs" FROM Procedure_T;
SELECT MAX(UnitCost) "Most Expensive Procedure" FROM Procedure_T;
SELECT MIN(UnitCost) "Cheapest Procedure" FROM Procedure_T;
select DISTINCT patientphonenum, InsPhoneNmbr from patient_t, insurance_t
UPDATE insurance_t SET insphonenmbr = 55512345678 WHERE insphonenmbr = 8002134567;
select DISTINCT insurance_t.insphonenmbr, insurance_t.inscompname from insurance_t, patient_t where patient_t.patientphonenum = insurance_t.insphonenmbr;
