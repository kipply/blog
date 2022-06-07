import React from 'react';
import { Row, Col } from 'react-bootstrap';

import meImg from './img/me.jpg';
import meImg2 from './img/me2.jpg';
import meImg3 from './img/me3.jpg';
import meImg4 from './img/me4.jpg';
import meImg5 from './img/me5.jpg';

const meImages = [meImg, meImg2, meImg3, meImg4, meImg5];

const Landing = () => (
  <div style={{ display: 'block' }}>
    <section className="panels" id="home" style={{ overflow: 'hidden' }}>
      <Row className="landing-row">
        <Col sm={4} md={4}>
          <img src={meImages[Math.floor(Math.random() * meImages.length)]} className="img-responsive img-circle" alt="kipply" />
        </Col>
        <Col sm={6} md={6} style={{ paddingTop: '20px' }}>
          <h2 className="section-heading">hi i&apos;m kipply (carol)</h2>
          <div id="links">
            <a href="/blog" target="_blank" rel="noredirect no referrer noreferrer">very good blog</a>
            <br />
            <a href="mailto:hello@carolchen.me" target="_blank" rel="noredirect no referrer noreferrer">email me</a>
            <br />
            <a href="https://twitter.com/kipperrii" target="_blank" rel="noredirect no referrer noreferrer">tweets (tweet at me?)</a>
          </div>
        </Col>
        <Col xs={12} sm={12} md={12} id="intro-text">
          <p>
            I am currently funemployed until I start at
            <a href="https://www.anthropic.com/" target="_blank" rel="noredirect no referrer noreferrer"> Anthropic</a>
            .
          </p>
          <p>
            Most of my thoughts are about my friends, effective altruism, my bunny,
            <a href="/blog/transformer-inference-arithmetic/" target="_blank" rel="noredirect no referrer noreferrer"> AI performance </a>
            (speed, not quality), and various other programming knick-knacks.
          </p>
          <p>
            I have been slowly learning to read, and I plan on learning to drive cars, do
            live-coding and solve all the New York Times crossword puzzles! I also enjoy
            pole dance + aerial arts, dance dance revolution and writing for my blog.
            My current residence is in San Francisco, though I&apos;ve spent some time in New
            York City and grew up in Richmond Hill, Ontario and also lived in Toronto.
          </p>
          <p>
            You may know me from one of my previous lives, like when I worked at
            <a href="/blog/real-good-things-about-internships/" target="_blank" rel="noredirect no referrer noreferrer"> Shopify</a>
            <a href="https://shopify.engineering/optimizing-ruby-lazy-initialization-in-truffleruby-with-deoptimization" target="_blank" rel="noredirect no referrer noreferrer"> on TruffleRuby </a>
            and subsequently thought
            <a href="/blog/jits-intro/" target="_blank" rel="noredirect no referrer noreferrer"> a lot</a>
            <a href="/blog/jits-impls/" target="_blank" rel="noredirect no referrer noreferrer"> about</a>
            <a href="/blog/escape-analysis/" target="_blank" rel="noredirect no referrer noreferrer"> jit compilers</a>
            . Maybe from when I attended Hackathons and also organised a few of them.
            Some other lives you probably don&apos;t know me from include working on edtech at
            <a href="https://www.hatchcoding.com/" target="_blank" rel="noredirect no referrer noreferrer"> Hatch Coding</a>
            ,
            <a href="https://www.sugarlabs.org/" target="_blank" rel="noredirect no referrer noreferrer"> Sugar Labs </a>
            and teaching at Ski Lakeridge or from the small town of
            <a href="https://dmoj.ca/"> contest programming </a>
            where I learned to code.
          </p>
        </Col>
      </Row>
    </section>
  </div>
);


export default Landing;
