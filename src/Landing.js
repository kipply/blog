import React from 'react';
import { Row, Col } from 'react-bootstrap';

import meImg from './img/me.jpg';
import meImg2 from './img/me2.jpg';
import meImg3 from './img/me3.jpg';
import meImg4 from './img/me4.jpg';
import meImg5 from './img/me5.jpg';

const meImages = [meImg, meImg2, meImg3, meImg4, meImg5];

const Landing = () => (
  <div style={{display:'block'}}>
    <section className="panels" id="home" style={{overflow: 'hidden'}}>
      <Row className="landing-row">
        <Col sm={4} md={4} >
          <img src={meImages[Math.floor(Math.random() * meImages.length)]} className="img-responsive img-circle"/>
        </Col>
        <Col sm={8} md={8} style={{paddingLeft: '50px'}} id="intro-text">
          <h2 className="section-heading">hi i'm carol (kipply)</h2>
          <p>
            I am currently occupied at Cohere AI where I've been living since we were a team of six! We're building an NLP toolkit and I work on serving our transformer models.
            
            I mostly think about my friends, work, effective altruism, my bunny and non-work related programming (in order, mostly). 
            
            My favourite features about myself are my optimism and adoration for the world. 
          </p>
          <p>
            Help me out? I'm currently struggling with finding flows/motivation to write, paths/activation energy to becoming a better programmer, picking the right community to settle into (in SF) and working around ADHD.  
          </p>
          <p>
            My main hobby is socializing, never hesitate to reach out to me (preferably via email).
          </p>
        <div id="links">
          <h3 className="section-heading">links</h3>
          <a href="https://carolchen.me/blog" target="_blank" rel="noredirect no referrer">very good blog</a> // <a href="mailto:hello@carolchen.me" target="_blank" rel="noredirect no referrer">email me</a> // <a href="https://twitter.com/kipperrii" target="_blank" rel="noredirect no referrer">tweets</a> // <a href="https://github.com/kipply" target="_blank" rel="noredirect no referrer">github</a> // <a href="https://carolchen.me/resume" target="_blank" rel="noredirect no referrer">button for recruiters</a>
        </div>
        </Col>
      </Row>
    </section>
  </div>
)


export default Landing;
