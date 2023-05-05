import BottomBar from "../commons/BottomBar"
import Contact_page from "../images/contact_page.jpg"

function About() {
    return (
        <div className="container">
            <img src={Contact_page} />
            <div className="container">
                <h1>Call us on 90 11 240!!</h1>
            </div>
            <div className="bottom-nav-bar">
                <BottomBar />
            </div>
        </div>
    );
}
export default About;