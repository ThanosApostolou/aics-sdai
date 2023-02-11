const { default: mongoose, Schema } = require('mongoose');

exports.Vehicle = mongoose.model('Vehicle', new Schema({
    brand: {
        type: String,
        required: true
    },
    type: {
        type: String,
        required: true
    },
    model: {
        type: String,
        required: true
    },
    year: {
        type: Number,
        required: true
    },
    engineCapacity: {
        type: String,
        required: true
    },
    enginePower: {
        type: String,
        required: true
    },
    photo: {
        type: String,
        required: false
    },
}));