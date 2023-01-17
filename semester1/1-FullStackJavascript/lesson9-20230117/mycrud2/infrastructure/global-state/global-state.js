class GlobalState {
    static instance;
    
    landmarks;
    
    constructor(landmarks) {
        this.landmarks = landmarks;
    }


    static initialize(landmarks) {
        GlobalState.instance = new GlobalState(landmarks);
    }

}

module.exports = {
    GlobalState
}