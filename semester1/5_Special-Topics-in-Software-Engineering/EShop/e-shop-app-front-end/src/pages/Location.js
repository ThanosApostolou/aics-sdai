import BottomBar from "../commons/BottomBar"

function Location() {
    return (
        <div className="container">
            <iframe title="location" src="https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d12585.750065684297!2d23.644353886079045!3d37.94356905502558!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x0%3A0x3e0dce8e58812705!2sUniversity%20of%20Piraeus!5e0!3m2!1sen!2sgr!4v1673549605091!5m2!1sen!2sgr"
                height="500" width="800" loading="lazy"></iframe>
            <div className="bottom-nav-bar">
                <BottomBar />
            </div>
        </div>
    );
}
export default Location;