////////
// Μόλις φορτώνεται η σελίδα εμφανίζεται δυναμικά ένα μήνυμα καλωσορίσματος για το χρήστη με μια συνάρτηση που καλείται μόνη της
////////
(function () {
    console.log('Welcome!')
    alert('Welcome!')
})();

////////
// Task 2: Register
////////
class Student {
    constructor(name, surname, avgScore) {
        this.name = name;
        this.surname = surname;
        this.avgScore = avgScore;
    }

    toHtml() {
        const studentLine = '' + this.name + ' '
            + this.surname + ' has average grade: '
            + this.avgScore;
        const studentHtml = '<p>' + studentLine + '</p>';
        return studentHtml;
    }
}

class FormModel {
    constructor(
        name,
        surname,
        numMaths,
        numCsharp,
        numJS,
        numJava
    ) {
        this.name = name;
        this.surname = surname;
        this.numMaths = numMaths;
        this.numCsharp = numCsharp;
        this.numJS = numJS;
        this.numJava = numJava;
    }

    validate() {
        const errors = [];
        if (!this.name) {
            errors.push('Name cannot be empty');
        }

        if (!this.surname) {
            errors.push('Surname cannot be empty');
        }

        if (this.numMaths == null || isNaN(this.numMaths) || typeof this.numMaths !== 'number') {
            errors.push('Maths must be a number');
        } else {
            if (this.numMaths < 0 || this.numMaths > 100) {
                errors.push('Maths must be between 0 and 100');
            }
        }

        if (this.numCsharp == null || isNaN(this.numCsharp) || typeof this.numCsharp !== 'number') {
            errors.push('C# must be a number');
        } else {
            if (this.numCsharp < 0 || this.numCsharp > 100) {
                errors.push('C# must be between 0 and 100');
            }
        }

        if (this.numJS == null || isNaN(this.numJS) || typeof this.numJS !== 'number') {
            errors.push('JavaScript must be a number');
        } else {
            if (this.numJS < 0 || this.numJS > 100) {
                errors.push('JavaScript must be between 0 and 100');
            }
        }

        if (this.numJava == null || isNaN(this.numJava) || typeof this.numJava !== 'number') {
            errors.push('Java must be a number');
        } else {
            if (this.numJava < 0 || this.numJava > 100) {
                errors.push('Java must be between 0 and 100');
            }
        }
        return errors;
    }

    static fromElements() {
        return new FormModel(
            document.getElementById('name').value,
            document.getElementById('surname').value,
            parseFloat(document.getElementById('numMaths').value),
            parseFloat(document.getElementById('numCsharp').value),
            parseFloat(document.getElementById('numJS').value),
            parseFloat(document.getElementById('numJava').value),
        );
    }
}

// all students
const students = []


// Πατώντας το κουμπί Register, μια συνάρτηση καλείται που
function register() {
    // Λαμβάνει το όνομα και το επώνυμο που εισήγαγε ο χρήστης
    const formModel = FormModel.fromElements();
    // Υπάρχει και ο ανάλογος έλεγχος για το αν είναι αριθμός ο βαθμός
    const errors = formModel.validate();
    if (errors && errors.length > 0) {
        let alertMsg = 'Form validation errors:\n';
        for (const error of errors) {
            alertMsg += '\t' + error + '\n';
        }
        alert(alertMsg);
        return;
    }

    // Καλεί μια άλλη συνάρτηση που υπολογίζει τον μέσο όρο των βαθμών
    const avgScore = calculateAvgScore(formModel.numMaths, formModel.numCsharp, formModel.numJS, formModel.numJava);
    // Δημιουργεί ένα νέο αντικείμενο χρησιμοποιώντας μια object constructor function:
    const newStudent = new Student(formModel.name, formModel.surname, avgScore);
    // Σε ένα array αποθηκεύει το νέο αντικείμενο (στο τέλος το array θα περιλαμβάνει όλα τα αντικείμενα που έχουν δημιουργηθεί)
    students.push(newStudent);

    // Αφού πατηθεί για πρώτη φορά το κουμπί Register, αλλάζει το κείμενό του και αντί για Register δείχνει Add another user
    // Θα μπορούσαμε να κάνουμε τον έλεγχο και με μια boolean
    if (students.length === 1) {
        document.getElementById('register').innerHTML = 'Add another user'
    }
}

function calculateAvgScore(numMaths, numCsharp, numJS, numJava) {
    const scoreSum = (numMaths ? numMaths : 0)
        + (numCsharp ? numCsharp : 0)
        + (numJS ? numJS : 0)
        + (numJava ? numJava : 0);
    const avgScore = scoreSum / 4;
    return avgScore;
}

// Πατώντας το κουμπί Display, μια συνάρτηση καλείται που εμφανίζει όλα τα αντικείμενα που έχουν δημιουργηθεί, για παράδειγμα
function display() {
    let innerHTML = '';
    for (const student of students) {
        innerHTML += student.toHtml();
    }
    document.getElementById('result').innerHTML = innerHTML;
}