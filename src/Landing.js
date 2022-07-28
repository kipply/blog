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
            <br />
            <a href="https://docs.google.com/spreadsheets/d/1s4OSqDGQWcC_oGHejdvd9839ASY0nKNGvWO3wQ0tgm4/edit#gid=0" target="_blank" rel="noredirect no referrer noreferrer">a spreadsheet</a>
          </div>
        </Col>
        <Col xs={12} sm={12} md={12} id="intro-text">
          <p>
            I am a generalist software engineer and my favourite colour is purple.
            This website changes colours when you refresh it!
          </p>
          <p>
            Most of my thoughts are about my friends, effective altruism, my bunny,
            AI (mostly non-research, like my
            <a href="/blog/transformer-inference-arithmetic/" target="_blank" rel="noredirect no referrer noreferrer"> thoughts on inference performance for large language models</a>
            )
            and various other programming knick-knacks.
          </p>
          <p>
            I have been slowly learning to read, and I plan on learning to drive cars, do
            live-coding and solve all the New York Times crossword puzzles! I also enjoy
            pole dance + aerial arts, dance dance revolution, writing for my blog and having
            other hobbies for no longer than two months at a time.
            My current residence is in San Francisco, though I&apos;ve spent some time in New
            York City and grew up in Richmond Hill, Ontario and also lived in Toronto.
          </p>
          <p>
            Where do I work? Read
            <a href="https://kipp.ly/blog/job-search-love-letters/"> my love letters </a>
            to find out. Someone said they read like a murder mystery!
            Before that, I was an early employee at Cohere for two and a half years
            where I worked across the board on building a platform to serve and train
            large language models.
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
