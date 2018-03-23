FROM ubuntu:16.04

RUN apt-get update && apt-get install -y ruby ruby-dev make build-essential nodejs npm
RUN gem install jekyll bundler jekyll-paginate jekyll-sitemap

EXPOSE 4000
CMD jekyll serve
